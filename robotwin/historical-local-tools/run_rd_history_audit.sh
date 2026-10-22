set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/scripts"
"$R/policy-venv/bin/python" -m unittest test_video_boundary test_history_reader > "$R/status/history-reader-tests.log" 2>&1
nohup bash -c 'flock -n "$1/status/history-reader-audit.lock" "$1/policy-venv/bin/python" "$1/scripts/audit_history_reader.py"; code=$?; echo "$code" > "$1/status/history-reader-audit.exitcode"' _ "$R" > "$R/status/history-reader-audit.log" 2>&1 < /dev/null &
echo $! > "$R/status/history-reader-audit.pid"