#!/usr/bin/env bash
set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec 9>"$R/status/r2-evaluation-exclusive.lock"
flock -n 9 || exit 3
# Hold legacy wrapper lock too; old final50 must never run alongside eval150.
exec 7>"$R/status/r2-final50-pair.lock"
flock -n 7 || exit 3
for task in cover_blocks plug_in_charger; do
 for seed in 0 1 2; do
  bash "$R/scripts/launch_r2_eval150.sh" "$task" "$seed" "${1:-}"
 done
done
"$R/policy-venv/bin/python" "$R/scripts/aggregate_r2_eval150.py"
