B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/verify_flux_final.py" > "$B/status/verify-flux-final.log" 2>&1 < /dev/null &
echo $! > "$B/status/verify-flux-final.pid"
