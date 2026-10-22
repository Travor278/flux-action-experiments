R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/scripts:$R/flux-action/src" PYTHONNOUSERSITE=1
nohup "$R/policy-venv/bin/python" "$R/scripts/audit_user_restart.py" > "$R/status/user-restart-audit-20261014.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$R/status/user-restart-audit-20261014.pid"
