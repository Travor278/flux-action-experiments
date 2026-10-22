set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
ulimit -c 0
nohup timeout 120 "$B/sim-venv/bin/python" "$B/scripts/sim_render_smoke.py" > "$B/status/sim-render-smoke.log" 2>&1 < /dev/null &
echo $! > "$B/status/sim-render-smoke.pid"
