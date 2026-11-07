#!/bin/sh
# A rented machine's entry for every command the controller runs: get uv, with
# which the agent gets its Python, once; then hand the command to the agent.
set -eu
home="${RENTAL_HOME:-$HOME/.rental}"
uv="$home/uv/uv"
if [ ! -x "$uv" ]; then
  if ! command -v curl >/dev/null 2>&1; then
    apt-get update -qq >/dev/null && apt-get install -y -qq curl ca-certificates >/dev/null
  fi
  curl -LsSf https://astral.sh/uv/0.13.4/install.sh |
    env UV_INSTALL_DIR="$home/uv" UV_NO_MODIFY_PATH=1 sh >"$home/uv-install.log" 2>&1
fi
PATH="$home/uv:$PATH" exec "$uv" run --quiet --no-project --python 3.12 python "$home/agent.py" "$@"
