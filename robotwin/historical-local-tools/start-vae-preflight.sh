set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
test "$free" -ge 30000
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
nohup timeout 600 "$B/flux-action/.venv/bin/python" "$B/scripts/vae_preflight.py" > "$B/status/vae-preflight.log" 2>&1 < /dev/null &
echo $! > "$B/status/vae-preflight.pid"
