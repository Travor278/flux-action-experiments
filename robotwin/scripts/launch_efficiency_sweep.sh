#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
export PATH="$B/flux-action/.venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$B/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"$B/status/efficiency-sweep.lock"
flock -n 9 || exit 3
python -c 'import torch;assert torch.cuda.device_count()==2 and all("H100" in torch.cuda.get_device_name(i) for i in range(2))'
python scripts/run_efficiency_sweep.py 2>&1 | tee status/efficiency-sweep-launch.log
