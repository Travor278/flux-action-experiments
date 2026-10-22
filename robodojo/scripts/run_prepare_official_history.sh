set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup bash -c '"$1/policy-venv/bin/python" "$1/scripts/prepare_official_history.py"; rc=$?; printf "%s\n" "$rc" > "$1/status/official-history-prepare.exitcode"' _ "$R" > "$R/status/official-history-prepare.log" 2>&1 < /dev/null &
