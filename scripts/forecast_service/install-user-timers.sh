#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$script_dir/../.." && pwd)"
destination="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
test -x "$repo/.venv/bin/python"

for job in epgnn-forecast epgnn-outcomes; do
    unit="$destination/$job.service"
    timer="$destination/$job.timer"
    if [[ -e "$unit" ]] && ! cmp -s "$unit" <(sed "s|@REPO@|$repo|g" "$script_dir/systemd/$job.service.in"); then
        printf 'Refusing to overwrite existing unit %s\n' "$unit" >&2
        exit 1
    fi
    if [[ -e "$timer" ]] && ! cmp -s "$timer" "$script_dir/systemd/$job.timer"; then
        printf 'Refusing to overwrite existing timer %s\n' "$timer" >&2
        exit 1
    fi
done

mkdir -p "$destination"
for job in epgnn-forecast epgnn-outcomes; do
    sed "s|@REPO@|$repo|g" "$script_dir/systemd/$job.service.in" > "$destination/$job.service"
    install -m 0644 "$script_dir/systemd/$job.timer" "$destination/$job.timer"
done

systemd-analyze --user verify "$destination/epgnn-forecast.timer" "$destination/epgnn-outcomes.timer"
systemctl --user daemon-reload
systemctl --user enable --now epgnn-forecast.timer epgnn-outcomes.timer
systemctl --user list-timers --all 'epgnn-*'
