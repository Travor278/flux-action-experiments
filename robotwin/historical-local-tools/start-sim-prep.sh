set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup env HF_XET_CACHE=/tmp/flux-piperx-eval-xet "$B/flux-action/.venv/bin/python" "$B/scripts/download_eval_assets.py" > "$B/status/eval-assets-download.log" 2>&1 < /dev/null &
echo $! > "$B/status/eval-assets-download.pid"
nohup bash "$B/scripts/bootstrap_sim.sh" > "$B/status/sim-bootstrap.log" 2>&1 < /dev/null &
echo $! > "$B/status/sim-bootstrap.pid"
