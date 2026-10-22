set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
bash -n "$R/scripts/launch_train_official_r2.sh"
bash -n "$R/scripts/launch_r2_final50.sh"
bash -n "$R/scripts/launch_r2_final50_pair.sh"
"$R/policy-venv/bin/python" "$R/scripts/prepare_r2_final50_plan.py" > "$R/status/r2-final50-prepare.log" 2>&1
"$R/policy-venv/bin/python" "$R/scripts/monitor_official_r2.py" > "$R/status/r2-monitor-refresh.log" 2>&1
