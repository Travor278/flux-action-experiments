R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup "$R/policy-venv/bin/python" "$R/scripts/cleanup_completed_preflight.py" > "$R/status/completed-preflight-cleanup.log" 2>&1 </dev/null &
