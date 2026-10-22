set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup "$R/policy-venv/bin/python" "$R/scripts/audit_official_switch_stop.py" > "$R/status/official-switch-stop-audit.log" 2>&1 < /dev/null &
