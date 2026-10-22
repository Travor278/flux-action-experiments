set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup bash -c 'flock -n "$1/status/prefix-coverage.lock" "$1/policy-venv/bin/python" "$1/scripts/audit_prefix_coverage.py"; code=$?; echo "$code" > "$1/status/prefix-coverage.exitcode"' _ "$R" > "$R/status/prefix-coverage.log" 2>&1 < /dev/null &
echo $! > "$R/status/prefix-coverage.pid"