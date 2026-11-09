#!/usr/bin/env bash
# Put bin/job-runner on PATH as job-runner.
set -euo pipefail

source="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bin/job-runner"
link="$HOME/.local/bin/job-runner"
mkdir -p "$(dirname "$link")"
if [[ -L "$link" && "$(readlink "$link")" == "$source" ]]; then
  :
elif [[ -e "$link" || -L "$link" ]]; then
  echo "skipped $link: something else is already there" >&2
  exit 1
else
  ln -s "$source" "$link"
  echo "linked $link"
fi
