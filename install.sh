#!/usr/bin/env bash
# Put bin/job-runner on PATH as job-runner, and run this workstation's watchers, which send each provider's machines
# back at their deadlines, as systemd user services that start with the workstation and come back if they stop: one
# for each provider whose key .env, beside this script, gives. Run it again after .env or the code changes, to
# restart them.
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

unit="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/job-runner-watch@.service"
mkdir -p "$(dirname "$unit")"
cat > "$unit.tmp" <<EOF
[Unit]
Description=Send back the machines this workstation rented from %i at their deadlines

[Service]
ExecStart=$(command -v python3) $source %i watch
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
mv "$unit.tmp" "$unit"
systemctl --user daemon-reload
for watcher in vast:VAST_API_KEY hetzner:HCLOUD_TOKEN runpod:RUNPOD_API_KEY; do
  provider="${watcher%%:*}"
  key="${watcher#*:}"
  service="job-runner-watch@$provider.service"
  if grep -q "^$key=." "$repo/.env" 2>/dev/null; then
    systemctl --user enable --quiet "$service"
    systemctl --user restart "$service"
    echo "the watcher of $provider runs as $service"
  else
    systemctl --user disable --quiet --now "$service" 2>/dev/null || true
    echo "no watcher of $provider: .env gives no $key"
  fi
done
if [[ "$(loginctl show-user "$USER" --property=Linger --value 2>/dev/null)" != "yes" ]]; then
  echo "the watchers stop with your last session until you run: loginctl enable-linger $USER" >&2
fi
