B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 "$B/scripts/cleanup-unused.py" > "$B/status/cleanup-unused.log" 2>&1
python3 "$B/scripts/eta.py"
