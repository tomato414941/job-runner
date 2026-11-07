"""Cloudflare R2 through S3's API, signed with AWS Signature Version 4, by the standard library alone.

The controller's key comes from the environment as AWS's tools take it:
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, and for a temporary
key AWS_SESSION_TOKEN and AWS_CREDENTIAL_EXPIRATION, when it expires in ISO 8601;
the bucket comes from the spec. A machine never sees the key, only URLs
presigned with it, which stop working when the key expires.
"""

import hashlib
import hmac
import os
import time
import xml.etree.ElementTree as ElementTree
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

# One PUT takes at most this; R2 keeps a presigned URL good for at most a week.
MAX_PUT = 5 * 1024**3
MAX_SECONDS = 7 * 24 * 3600
EMPTY = hashlib.sha256(b"").hexdigest()


def encode(text, safe=""):
    return quote(text, safe=safe)


def signing_key(secret, date, region):
    key = ("AWS4" + secret).encode()
    for part in (date, region, "s3", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return key


def signature(credentials, region, moment, canonical):
    """Sign a canonical request; moment is a time.struct_time in UTC."""
    date, stamp = time.strftime("%Y%m%d", moment), time.strftime("%Y%m%dT%H%M%SZ", moment)
    scope = f"{date}/{region}/s3/aws4_request"
    text = "\n".join(["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    return hmac.new(signing_key(credentials["secret"], date, region), text.encode(), hashlib.sha256).hexdigest()


def presign(credentials, region, moment, method, host, path, seconds, query=None):
    """A URL that lets its holder make one request until it expires: query-string SigV4 over the host alone."""
    date, stamp = time.strftime("%Y%m%d", moment), time.strftime("%Y%m%dT%H%M%SZ", moment)
    parameters = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{credentials['id']}/{date}/{region}/s3/aws4_request",
        "X-Amz-Date": stamp, "X-Amz-Expires": str(int(min(seconds, MAX_SECONDS))), "X-Amz-SignedHeaders": "host",
        **({"X-Amz-Security-Token": credentials["token"]} if credentials.get("token") else {}),
        **{key: str(value) for key, value in (query or {}).items()},
    }
    query_text = "&".join(f"{encode(key)}={encode(value)}" for key, value in sorted(parameters.items()))
    canonical = "\n".join([method, path, query_text, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD"])
    return f"{path}?{query_text}&X-Amz-Signature={signature(credentials, region, moment, canonical)}"


def authorization(credentials, region, moment, method, host, path, query, headers, payload_hash):
    """The headers that sign a request: SigV4 over its host, its headers and its payload's hash."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", moment)
    signed = {key.lower(): str(value).strip() for key, value in headers.items()}
    signed.update({"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": stamp})
    if credentials.get("token"):
        signed["x-amz-security-token"] = credentials["token"]
    names = sorted(signed)
    query_text = "&".join(f"{encode(key)}={encode(str(value))}" for key, value in sorted((query or {}).items()))
    canonical = "\n".join([method, path, query_text, "".join(f"{name}:{signed[name]}\n" for name in names),
                           ";".join(names), payload_hash])
    credential = f"{credentials['id']}/{time.strftime('%Y%m%d', moment)}/{region}/s3/aws4_request"
    return {**signed, "authorization": f"AWS4-HMAC-SHA256 Credential={credential},SignedHeaders={';'.join(names)},"
                                       f"Signature={signature(credentials, region, moment, canonical)}"}


class R2Store:
    max_put = MAX_PUT

    def __init__(self, bucket, environment=None, clock=time.time):
        environment = os.environ if environment is None else environment
        self.bucket, self.clock = bucket, clock
        self.endpoint = environment["AWS_ENDPOINT_URL"].rstrip("/")
        self.host = urlsplit(self.endpoint).netloc
        self.region = environment.get("AWS_REGION", "auto")
        self.credentials = {"id": environment["AWS_ACCESS_KEY_ID"], "secret": environment["AWS_SECRET_ACCESS_KEY"],
                            "token": environment.get("AWS_SESSION_TOKEN")}
        expiration = environment.get("AWS_CREDENTIAL_EXPIRATION")
        self.expires = datetime.fromisoformat(expiration).timestamp() if expiration else None

    def path(self, key):
        return f"/{encode(self.bucket)}/{encode(key, safe='/')}"

    def presign(self, method, key, seconds, query=None):
        moment = time.gmtime(self.clock())
        return self.endpoint + presign(self.credentials, self.region, moment, method, self.host, self.path(key),
                                       seconds, query)

    def request(self, method, key, query=None, body=b"", headers=None, payload_hash=None):
        """Make a signed request; its status, headers and body, with 404 answered rather than raised."""
        payload_hash = payload_hash or (hashlib.sha256(body).hexdigest() if isinstance(body, bytes) else EMPTY)
        signed = authorization(self.credentials, self.region, time.gmtime(self.clock()), method, self.host,
                               self.path(key), query, headers or {}, payload_hash)
        query_text = "&".join(f"{encode(name)}={encode(str(value))}" for name, value in sorted((query or {}).items()))
        url = self.endpoint + self.path(key) + (f"?{query_text}" if query_text else "")
        signed.pop("host")
        try:
            with urlopen(Request(url, data=body if body else None, method=method, headers=signed),
                         timeout=600) as response:
                return response.status, response.headers, response.read()
        except HTTPError as error:
            if error.code == 404:
                return 404, error.headers, b""
            raise

    def head(self, key):
        status, headers, _ = self.request("HEAD", key)
        return None if status == 404 else headers

    def size(self, key):
        headers = self.head(key)
        return None if headers is None else int(headers["Content-Length"])

    def etag(self, key):
        headers = self.head(key)
        return None if headers is None else headers["ETag"].strip('"')

    def read(self, key):
        status, _, body = self.request("GET", key)
        return None if status == 404 else body

    def write(self, key, data):
        self.request("PUT", key, body=data, headers={"content-length": str(len(data))})

    def start_upload(self, key):
        _, _, body = self.request("POST", key, {"uploads": ""})
        return ElementTree.fromstring(body).find("{*}UploadId").text

    def finish_upload(self, key, upload, etags):
        parts = "".join(f"<Part><PartNumber>{number}</PartNumber><ETag>\"{etag}\"</ETag></Part>"
                        for number, etag in etags)
        body = f"<CompleteMultipartUpload>{parts}</CompleteMultipartUpload>".encode()
        _, _, answer = self.request("POST", key, {"uploadId": upload}, body)
        # S3 may answer 200 with an error in the body.
        if ElementTree.fromstring(answer).tag.endswith("Error"):
            raise ValueError("the store did not finish the upload")

    def upload(self, key, path, part_bytes=1 << 30):
        """Send a file of this machine's to key, in parts when one PUT cannot take it."""
        size = os.path.getsize(path)
        with open(path, "rb") as source:
            if size <= self.max_put:
                self.request("PUT", key, body=source.read(), headers={"content-length": str(size)})
                return
            upload, etags = self.start_upload(key), []
            for number, start in enumerate(range(0, size, part_bytes), 1):
                source.seek(start)
                chunk = source.read(part_bytes)
                _, headers, _ = self.request("PUT", key, {"partNumber": number, "uploadId": upload}, chunk,
                                             {"content-length": str(len(chunk))})
                etags.append([number, headers["ETag"].strip('"')])
            self.finish_upload(key, upload, etags)
