#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
TASK=${1:?expected click_bell or handover_block}
case "$TASK" in click_bell|handover_block) ;; *) exit 2;; esac
test -s "$B/status/pi05-data-verified.json"
test -s "$B/status/pi05-base-forward.json"
export PYTHONPATH="$B/flux-action/src:$B/scripts:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE=/tmp/flux-piperx-pi05-datasets
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=/opt/a2r/venv/bin/python
"$PY" -c 'import torch;assert torch.cuda.device_count()==2;assert all("H100" in torch.cuda.get_device_name(i) for i in range(2));print("PI05_H100_PREFLIGHT",torch.__version__,flush=True)'
STATUS="$B/status/pi05_${TASK}_seed42"
mkdir -p "$STATUS"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 15 > "$STATUS/gpu.csv" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 "$B/scripts/train_pi05.py" "$TASK" --steps 3000 2>&1 | tee -a "$STATUS/train.log"
test -s "$STATUS/COMPLETE.json"
