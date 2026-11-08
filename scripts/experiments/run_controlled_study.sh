#!/usr/bin/env bash
# Start this named launcher inside a persistent host tmux session.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
mkdir -p runs/controlled_study_launch
exec 9>runs/controlled_study_launch/launcher.lock
if ! flock -n 9; then
    echo 'Another controlled-study launcher is already running.' >&2
    exit 1
fi
log=runs/controlled_study_launch/full.log
finish() {
    result=$?
    printf '{"exit_code": %s}\n' "$result" >runs/controlled_study_launch/exit.json.tmp
    mv runs/controlled_study_launch/exit.json.tmp runs/controlled_study_launch/exit.json
}
trap finish EXIT
rm -f runs/controlled_study_launch/exit.json
printf '\nControlled study started at %s\n' "$(date --iso-8601=seconds)" | tee -a "$log"
.venv/bin/python -u -m epgnn.controlled_study full "$@" 2>&1 | tee -a "$log"
printf 'Controlled study completed at %s\n' "$(date --iso-8601=seconds)" | tee -a "$log"
