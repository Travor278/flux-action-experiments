set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
source "$B/scripts/sim-env.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 XDG_RUNTIME_DIR=/tmp/flux-piperx-policy-chunk-runtime
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
test "$free" -gt 12288
nohup flock -n "$B/status/policy-chunk-replay.lock" "$B/sim-venv/bin/python" "$B/scripts/replay_policy_chunks.py" > "$B/status/policy-chunk-replay.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/policy-chunk-replay.pid"
