"""Measure a rented machine in about a minute and print one JSON object: what nvidia-smi reports of each GPU, how fast
each GPU multiplies bfloat16 matrices and copies within its memory, its SM clock and power draw while it multiplies,
how far its products of a fixed pair of matrices stray from the CPU's and from each other, how fast the host copies to
the first GPU, how fast the machine downloads from two public sources and writes to its disk, and the country that its
traffic leaves from.

It runs with the Python of the machine's image, which needs PyTorch for the GPUs' speeds: python3 - < machine_check.py
An image without PyTorch but with nvcc, as CUDA's devel images have, still gets the products checked, by a small
cuBLAS program that nvcc builds on the machine. A change to what or how it measures changes VERSION, since only checks
of one version are compared.
"""

import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
import urllib.request

VERSION = 3
QUERY = ("name", "memory.used", "memory.total", "power.limit", "power.default_limit", "power.max_limit",
         "clocks.max.sm", "pcie.link.gen.max", "pcie.link.width.max", "temperature.gpu", "driver_version")
KEYS = ("name", "memory_used_mib", "memory_total_mib", "power_limit_w", "power_default_w", "power_max_w",
        "sm_clock_max_mhz", "pcie_gen", "pcie_width", "temperature_c", "driver")
TEXT = {"name", "driver"}
MATRIX = 8192
# The side of the fixed matrices whose products the CPU works out too; a GPU that answers them wrongly, or differently
# twice, gives any work wrong numbers whatever its speed.
ANSWER_MATRIX = 1024
# A stand-in for a product that is not finite, which JSON cannot carry.
NOT_FINITE = 1e30
ANSWER_NAMES = ("float32_error", "float32_repeat_gap", "bfloat16_error", "bfloat16_repeat_gap")
# The same check as `answers`, for an image without PyTorch: cuBLAS multiplies a fixed pair of 512 x 512 matrices on
# each GPU, in float32 and with bfloat16 inputs, twice each, and the program prints how far the products stray from the
# CPU's float64 products and from each other, relative to the largest entry, as a JSON list with one object a GPU.
ANSWERS_CU = r"""
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

static const int N = 512;
static const double NOT_FINITE = 1e30;

static float next_value(unsigned long long &state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)(state >> 40) / (double)(1ULL << 24) * 2.0 - 1.0);
}

static std::vector<double> reference(const std::vector<float> &left, const std::vector<float> &right) {
    std::vector<double> product((size_t)N * N, 0.0);
    for (int column = 0; column < N; ++column)
        for (int inner = 0; inner < N; ++inner) {
            double factor = right[inner + (size_t)column * N];
            for (int row = 0; row < N; ++row)
                product[row + (size_t)column * N] += (double)left[row + (size_t)inner * N] * factor;
        }
    return product;
}

static double share(double found, double scale) {
    double value = found / scale;
    return std::isfinite(value) ? value : NOT_FINITE;
}

static double stray(const std::vector<float> &got, const std::vector<double> &expected, double scale) {
    double found = 0.0;
    for (size_t index = 0; index < got.size(); ++index) {
        double difference = std::fabs((double)got[index] - expected[index]);
        if (!std::isfinite(difference)) return NOT_FINITE;
        found = std::fmax(found, difference);
    }
    return share(found, scale);
}

static double gap(const std::vector<float> &first, const std::vector<float> &second, double scale) {
    double found = 0.0;
    for (size_t index = 0; index < first.size(); ++index) {
        double difference = std::fabs((double)first[index] - (double)second[index]);
        if (!std::isfinite(difference)) return NOT_FINITE;
        found = std::fmax(found, difference);
    }
    return share(found, scale);
}

#define CHECK(call) do { if ((int)(call) != 0) { std::printf("{\"answers_error\": \"%s failed\"}\n", #call); return 0; } } while (0)

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::printf("{\"answers_error\": \"no CUDA device\"}\n");
        return 0;
    }
    size_t count = (size_t)N * N;
    std::vector<float> left(count), right(count), left16(count), right16(count);
    std::vector<__nv_bfloat16> left_bf(count), right_bf(count);
    unsigned long long state = 1;
    for (float &value : left) value = next_value(state);
    for (float &value : right) value = next_value(state);
    for (size_t index = 0; index < count; ++index) {
        left_bf[index] = __float2bfloat16(left[index]);
        right_bf[index] = __float2bfloat16(right[index]);
        left16[index] = __bfloat162float(left_bf[index]);
        right16[index] = __bfloat162float(right_bf[index]);
    }
    std::vector<double> expected32 = reference(left, right), expected16 = reference(left16, right16);
    double scale32 = 0.0, scale16 = 0.0;
    for (double value : expected32) scale32 = std::fmax(scale32, std::fabs(value));
    for (double value : expected16) scale16 = std::fmax(scale16, std::fabs(value));
    std::string rows;
    for (int device = 0; device < devices; ++device) {
        CHECK(cudaSetDevice(device));
        cublasHandle_t handle;
        CHECK(cublasCreate(&handle));
        float *left32_d, *right32_d, *product_d;
        __nv_bfloat16 *left16_d, *right16_d;
        CHECK(cudaMalloc((void **)&left32_d, count * sizeof(float)));
        CHECK(cudaMalloc((void **)&right32_d, count * sizeof(float)));
        CHECK(cudaMalloc((void **)&product_d, count * sizeof(float)));
        CHECK(cudaMalloc((void **)&left16_d, count * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc((void **)&right16_d, count * sizeof(__nv_bfloat16)));
        CHECK(cudaMemcpy(left32_d, left.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(right32_d, right.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(left16_d, left_bf.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(right16_d, right_bf.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        const float one = 1.0f, zero = 0.0f;
        std::vector<float> first32(count), second32(count), first16(count), second16(count);
        for (int run = 0; run < 2; ++run) {
            CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, left32_d, N, right32_d, N, &zero,
                              product_d, N));
            CHECK(cudaMemcpy((run ? second32 : first32).data(), product_d, count * sizeof(float),
                             cudaMemcpyDeviceToHost));
        }
        for (int run = 0; run < 2; ++run) {
            CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, left16_d, CUDA_R_16BF, N, right16_d,
                               CUDA_R_16BF, N, &zero, product_d, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT));
            CHECK(cudaMemcpy((run ? second16 : first16).data(), product_d, count * sizeof(float),
                             cudaMemcpyDeviceToHost));
        }
        char row[256];
        std::snprintf(row, sizeof row,
                      "%s{\"float32_error\": %.6g, \"float32_repeat_gap\": %.6g, \"bfloat16_error\": %.6g, "
                      "\"bfloat16_repeat_gap\": %.6g}", device ? ", " : "",
                      stray(first32, expected32, scale32), gap(first32, second32, scale32),
                      stray(first16, expected16, scale16), gap(first16, second16, scale16));
        rows += row;
        cudaFree(left32_d); cudaFree(right32_d); cudaFree(product_d); cudaFree(left16_d); cudaFree(right16_d);
        cublasDestroy(handle);
    }
    std::printf("[%s]\n", rows.c_str());
    return 0;
}
"""
# Two public sources on different networks, so that one source's outage does not look like a slow machine. At most the
# first 512 MiB of each are timed, so that hosts that charge for traffic charge little.
DOWNLOADS = {
    "huggingface": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet",
    # Cloudflare refuses 100 MB or more at once.
    "cloudflare": "https://speed.cloudflare.com/__down?bytes=99999999",
}
DOWNLOAD_BYTES = 512 * 2**20
DOWNLOAD_SECONDS = 15
# Cloudflare's answer names the country that the machine's traffic leaves from, which a listing can misstate.
TRACE = "https://www.cloudflare.com/cdn-cgi/trace"


def number(text):
    try:
        return float(text)
    except ValueError:
        return None


def parse_gpus(text):
    """nvidia-smi's rows, without header or units, as one dictionary a GPU; a value it lacks, like "[N/A]", is None."""
    rows = []
    for line in text.splitlines():
        if line.strip():
            values = [value.strip() for value in line.split(",")]
            rows.append({key: value if key in TEXT else number(value) for key, value in zip(KEYS, values)})
    return rows


def gpus():
    done = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(QUERY), "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=60)
    return parse_gpus(done.stdout)


def parse_samples(text):
    """The medians of the SM clock and of the power draw in nvidia-smi's samples of the two."""
    rows = [[number(value.strip()) for value in line.split(",")] for line in text.splitlines() if line.count(",") == 1]
    clocks = [clock for clock, _ in rows if clock]
    draws = [draw for _, draw in rows if draw]
    return (statistics.median(clocks) if clocks else None, statistics.median(draws) if draws else None)


def rate(step, seconds, synchronize):
    """How many times a second `step` runs, timed over at least `seconds` after one untimed run."""
    step()
    synchronize()
    count, began = 0, time.perf_counter()
    while True:
        for _ in range(5):
            step()
        count += 5
        synchronize()
        elapsed = time.perf_counter() - began
        if elapsed >= seconds:
            return count / elapsed


def stray(value, scale):
    """`value` relative to `scale`, or NOT_FINITE when it is not a finite number."""
    share = value / scale
    return share if math.isfinite(share) else NOT_FINITE


def answers(torch, device):
    """How far the GPU's products of a fixed pair of matrices stray from the CPU's float64 products, in float32 and in
    bfloat16, and how far two runs of each differ, both relative to the largest entry of the CPU's product."""
    generator = torch.Generator().manual_seed(0)
    left, right = (torch.randn(ANSWER_MATRIX, ANSWER_MATRIX, generator=generator) for _ in range(2))
    found = {}
    for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        cpu_left, cpu_right = left.to(dtype), right.to(dtype)
        expected = cpu_left.double() @ cpu_right.double()
        scale = expected.abs().max().item()
        gpu_left, gpu_right = cpu_left.to(device), cpu_right.to(device)
        first, second = ((gpu_left @ gpu_right).double().cpu() for _ in range(2))
        found[f"{name}_error"] = stray((first - expected).abs().max().item(), scale)
        found[f"{name}_repeat_gap"] = stray((first - second).abs().max().item(), scale)
    return found


def gpu_speeds(count):
    """Each GPU's speeds and answers, the slowest GPU's speeds as the machine's, since training on several GPUs waits
    for the slowest, and the worst GPU's answers as the machine's."""
    import torch

    found = {"torch": torch.__version__, "cuda": torch.version.cuda, "each": []}
    for index in range(count):
        device = torch.device("cuda", index)

        def synchronize():
            torch.cuda.synchronize(device)

        left, right = (torch.randn(MATRIX, MATRIX, device=device, dtype=torch.bfloat16) for _ in range(2))
        product = torch.empty_like(left)
        sampler = subprocess.Popen(["nvidia-smi", "-i", str(index), "--query-gpu=clocks.sm,power.draw",
                                    "--format=csv,noheader,nounits", "-lms", "500"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            products = rate(lambda: torch.mm(left, right, out=product), 6, synchronize)
        finally:
            sampler.terminate()
            samples, _ = sampler.communicate(timeout=30)
        clock, draw = parse_samples(samples)
        source = torch.empty(2**28, device=device)  # 1 GiB of float32
        target = torch.empty_like(source)
        copies = rate(lambda: target.copy_(source), 2, synchronize)
        found["each"].append({"matmul_tflops": products * 2 * MATRIX**3 / 1e12,
                              "memory_gbps": copies * 2 * source.numel() * source.element_size() / 1e9,
                              "sm_clock_under_load_mhz": clock, "power_under_load_w": draw})
        del left, right, product, source, target
        torch.cuda.empty_cache()
        # After the load, when a failing GPU is likeliest to show it.
        found["each"][-1].update(answers(torch, device))
    host = torch.empty(2**28, pin_memory=True)
    on_gpu = torch.empty(2**28, device="cuda:0")
    uploads = rate(lambda: on_gpu.copy_(host, non_blocking=True), 2, lambda: torch.cuda.synchronize(0))
    found["host_to_gpu_gbps"] = uploads * host.numel() * host.element_size() / 1e9
    for name in ("matmul_tflops", "memory_gbps", "sm_clock_under_load_mhz", "power_under_load_w"):
        values = [each[name] for each in found["each"] if each[name] is not None]
        found[name] = min(values) if values else None
    for name in ANSWER_NAMES:
        found[name] = max((each[name] for each in found["each"] if name in each), default=None)
    return found


def answers_by_nvcc():
    """For an image without PyTorch, the same measures as `answers` for each GPU, which ANSWERS_CU works out once nvcc
    has built it on the machine, and the worst GPU's as the machine's; None when the image has no nvcc."""
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    if nvcc is None:
        return None
    with tempfile.TemporaryDirectory() as folder:
        source, program = os.path.join(folder, "answers.cu"), os.path.join(folder, "answers")
        with open(source, "w") as handle:
            handle.write(ANSWERS_CU)
        built = subprocess.run([nvcc, "-O2", "-o", program, source, "-lcublas"], capture_output=True, text=True,
                               timeout=240)
        if built.returncode:
            return {"answers_error": ("nvcc failed: " + built.stderr)[-300:]}
        done = subprocess.run([program], capture_output=True, text=True, timeout=120)
    return read_answers(done.stdout)


def read_answers(text):
    """ANSWERS_CU's output as results: each GPU's answers, and the worst GPU's as the machine's."""
    lines = [line for line in text.splitlines() if line.startswith(("[", "{"))]
    found = json.loads(lines[-1]) if lines else {"answers_error": "the answers program printed nothing"}
    if isinstance(found, dict):
        return found
    return {"answers_by": "nvcc", "each_answers": found,
            **{name: max((each[name] for each in found), default=None) for name in ANSWER_NAMES}}


def download_mbps(url, size=DOWNLOAD_BYTES, seconds=DOWNLOAD_SECONDS):
    """MB/s over at most the first `size` bytes of `url` within about `seconds`; 0.0 if nothing arrived."""
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}", "User-Agent": "machine-check"})
    began, received = time.perf_counter(), 0
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            while received < size and time.perf_counter() - began < seconds:
                chunk = response.read(2**20)
                if not chunk:
                    break
                received += len(chunk)
    except Exception:  # noqa: BLE001 - a refused, broken or stalled download counts what arrived before it
        pass
    return received / (time.perf_counter() - began) / 1e6


def downloads(measure=download_mbps):
    """Each source's speed, and the faster one as the machine's: a slow source alone does not make a slow machine."""
    each = {name: round(measure(url), 1) for name, url in DOWNLOADS.items()}
    return each, max(each.values())


def parse_trace(text):
    """The country code in Cloudflare's trace, or None."""
    return next((line.split("=", 1)[1].strip() for line in text.splitlines() if line.startswith("loc=")), None)


def egress_country():
    try:
        with urllib.request.urlopen(urllib.request.Request(TRACE, headers={"User-Agent": "machine-check"}),
                                    timeout=20) as response:
            return parse_trace(response.read(4096).decode(errors="replace"))
    except Exception:  # noqa: BLE001 - a machine that cannot reach Cloudflare shows it in its downloads
        return None


def disk_write_mbps(directory, size=2**30):
    block = os.urandom(2**20) * 64
    with tempfile.NamedTemporaryFile(dir=directory) as stream:
        began = time.perf_counter()
        for _ in range(size // len(block)):
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
        return size / (time.perf_counter() - began) / 1e6


def host():
    found = {"cpus": len(os.sched_getaffinity(0))}
    try:
        with open("/proc/cpuinfo") as lines:
            found["cpu"] = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")), None)
        with open("/proc/meminfo") as lines:
            found["memory_gib"] = next((round(int(line.split()[1]) / 2**20, 1) for line in lines
                                        if line.startswith("MemTotal")), None)
    except OSError:
        pass
    return found


def main():
    began = time.time()
    results = {"version": VERSION, "gpus": gpus(), **host()}
    try:
        results.update(gpu_speeds(len(results["gpus"])))
    except ImportError as error:
        results["speed_error"] = f"{type(error).__name__}: {error}"[-300:]
        try:
            results.update(answers_by_nvcc() or {"answers_error": "the image has neither PyTorch nor nvcc"})
        except Exception as error:  # noqa: BLE001 - a machine whose products cannot be checked still reports the rest
            results["answers_error"] = f"{type(error).__name__}: {error}"[-300:]
    except Exception as error:  # noqa: BLE001 - a machine that cannot measure its GPUs still reports the rest
        results["speed_error"] = f"{type(error).__name__}: {error}"[-300:]
    results["downloads"], results["download_mbps"] = downloads()
    results["egress_country"] = egress_country()
    try:
        results["disk_write_mbps"] = disk_write_mbps(".")
    except OSError as error:
        results["disk_error"] = str(error)[-300:]
    results["seconds"] = round(time.time() - began, 1)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
