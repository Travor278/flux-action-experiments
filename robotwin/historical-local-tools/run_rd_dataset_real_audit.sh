set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/scripts"
nohup bash -c 'flock -n "$1/status/history-dataset-real-audit.lock" "$1/policy-venv/bin/python" "$1/scripts/audit_history_dataset.py"; code=$?; echo "$code" > "$1/status/history-dataset-real-audit.exitcode"' _ "$R" > "$R/status/history-dataset-real-audit.log" 2>&1 < /dev/null &
echo $! > "$R/status/history-dataset-real-audit.pid"