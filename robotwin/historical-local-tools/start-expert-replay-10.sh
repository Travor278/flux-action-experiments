set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
source "$B/scripts/sim-env.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 XDG_RUNTIME_DIR=/tmp/flux-piperx-expert-replay-runtime
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
"$B/sim-venv/bin/python" - <<'PY'
import subprocess,pathlib,json,time
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert free>12288,free
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');(B/'status/expert-replay-resource-preflight.json').write_text(json.dumps({'time':time.time(),'free_mib':free,'minimum_mib':12288,'torch_allocator_fraction_cap':.12,'no_second_model':True}))
PY
nohup flock -n "$B/status/expert-replay.lock" "$B/sim-venv/bin/python" "$B/scripts/run_expert_replay.py" --limit 10 > "$B/status/expert-replay-batch10.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/expert-replay.pid"
