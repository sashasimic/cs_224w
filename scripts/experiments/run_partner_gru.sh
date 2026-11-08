#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
mkdir -p runs/partner_gru_launch
exec 9>runs/partner_gru_launch/launcher.lock
if ! flock -n 9; then
    echo 'Another partner-GRU launcher is running.' >&2
    exit 1
fi
log=runs/partner_gru_launch/full.log
finish() {
    code=$?
    printf '{"exit_code":%s}\n' "$code" >runs/partner_gru_launch/exit.json.tmp
    mv runs/partner_gru_launch/exit.json.tmp runs/partner_gru_launch/exit.json
}
trap finish EXIT
printf '{"state":"running"}\n' >runs/partner_gru_launch/exit.json
printf '\nPartner-style GRU started at %s\n' "$(date --iso-8601=seconds)" | tee -a "$log"
.venv/bin/python -u -m epgnn.partner_gru full "$@" 2>&1 | tee -a "$log"
printf 'Partner-style GRU finished at %s\n' "$(date --iso-8601=seconds)" | tee -a "$log"
