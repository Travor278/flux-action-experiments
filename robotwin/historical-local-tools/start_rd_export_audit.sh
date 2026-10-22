set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup "$R/policy-venv/bin/python" "$R/scripts/audit_full_preflight_export.py" > "$R/status/full-preflight-export-audit.log" 2>&1 < /dev/null &
echo $! > "$R/status/full-preflight-export-audit.pid"