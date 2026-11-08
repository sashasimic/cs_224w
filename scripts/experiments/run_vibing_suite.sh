#!/usr/bin/env bash
# Run in the foreground inside tmux. Resume with this same command after a stop.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
mkdir -p runs/vibing_suite_launch
exec 9>runs/vibing_suite_launch/launcher.lock
if ! flock -n 9; then
    echo 'Another full-suite launcher is already running.' >&2
    exit 1
fi
log=runs/vibing_suite_launch/full.log
finish() {
    result=$?
    printf '{"exit_code": %s}\n' "$result" >runs/vibing_suite_launch/exit.json.tmp
    mv runs/vibing_suite_launch/exit.json.tmp runs/vibing_suite_launch/exit.json
}
trap finish EXIT
rm -f runs/vibing_suite_launch/exit.json
printf '\nFull suite started at %s\n' "$(date --iso-8601=seconds)" | tee -a "$log"
.venv/bin/python -u -m epgnn.experiment_suite full 2>&1 | tee -a "$log"
printf 'Full suite completed at %s\n' "$(date --iso-8601=seconds)" | tee -a "$log"
