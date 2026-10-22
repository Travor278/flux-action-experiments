R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup "$R/policy-venv/bin/python" "$R/scripts/audit_ws_and_cleanup_export.py" > "$R/status/ws-final-audit-cleanup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$R/status/ws-final-audit-cleanup.pid"
