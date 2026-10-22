R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup "$R/policy-venv/bin/python" "$R/scripts/cleanup_abandoned_prefix_20261021.py" > "$R/status/abandoned-prefix-cleanup-20261021.log" 2>&1 < /dev/null &
