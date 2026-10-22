set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
G=/workspace/hdd/codex_remote_ops/A2R_pickup_gen_20261007
for i in $(seq 1 120); do
 pid=$(cat "$B/status/expert-smoke.pid")
 if ! [ -r "/proc/$pid/cmdline" ] || ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$B/scripts/expert_smoke.py"; then break; fi
 sleep 5
done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 30000 ]; then echo 'DEFERRED: less than30GB free on notebook GPU'; exit 75; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE=/tmp/flux-piperx-pi05-datasets
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
exec timeout 1200 "$G/venv/bin/python" "$B/scripts/pi05_base_preflight.py"
