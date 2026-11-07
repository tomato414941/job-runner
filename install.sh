#!/usr/bin/env bash
# Put bin/rental on PATH as rental.
set -euo pipefail

source="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bin/rental"
link="$HOME/.local/bin/rental"
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
