#!/usr/bin/env bash
# Put bin/job-runner on PATH as job-runner, and run this workstation's watcher of Vast.ai machines as a systemd user
# service that starts with the workstation and comes back if it stops. Run it again after vast/machines.py changes,
# to restart the watcher.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source="$repo/bin/job-runner"
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

unit="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/job-runner-watch.service"
mkdir -p "$(dirname "$unit")"
cat > "$unit.tmp" <<EOF
[Unit]
Description=Send back this workstation's Vast.ai machines at their deadlines

[Service]
ExecStart=$(command -v python3) $repo/vast/machines.py watch
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
mv "$unit.tmp" "$unit"
systemctl --user daemon-reload
systemctl --user enable job-runner-watch.service
systemctl --user restart job-runner-watch.service
echo "the watcher runs as job-runner-watch.service"
if [[ "$(loginctl show-user "$USER" --property=Linger --value 2>/dev/null)" != "yes" ]]; then
  echo "the watcher stops with your last session until you run: loginctl enable-linger $USER" >&2
fi
