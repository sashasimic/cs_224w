#!/usr/bin/env bash
# Launch inside a persistent host tmux session after audit and smoke pass.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
mkdir -p runs/airport_temperature_launch
exec 9>runs/airport_temperature_launch/launcher.lock
if ! flock -n 9; then
    echo 'Another airport-temperature launcher is running.' >&2
    exit 1
fi
log=runs/airport_temperature_launch/full.log
finish() {
    result=$?
    printf '{"exit_code": %s}\n' "$result" >runs/airport_temperature_launch/exit.json.tmp
    mv runs/airport_temperature_launch/exit.json.tmp runs/airport_temperature_launch/exit.json
}
trap finish EXIT
printf '{"state":"running"}\n' >runs/airport_temperature_launch/exit.json
printf '\nAirport temperature study started at %s\n' "$(date --iso-8601=seconds)" | tee -a "$log"
.venv/bin/python -u -m epgnn.airport_study full "$@" 2>&1 | tee -a "$log"
printf 'Airport temperature study completed at %s\n' "$(date --iso-8601=seconds)" | tee -a "$log"
