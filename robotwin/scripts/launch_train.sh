#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
TASK=${1:?expected click_bell or handover_block}
case "$TASK" in click_bell|handover_block) ;; *) exit 2;; esac
cd "$B"
export PATH="$B/flux-action/.venv/bin:$PATH"
export PYTHONPATH="$B/flux-action/src:$B:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
test -f status/data-ready.txt
test -f assets/flux-base.manifest.json
test -s status/checkpoint-repair-verified.json
python -c 'import json; r=json.load(open("status/checkpoint-repair-verified.json")); assert r["save_resume_export"] and r["world_size"]==8'
read -r N MICRO ACCUM < <(python -c 'import json; s=json.load(open("status/profile-decision.json"))["selected"]; n=s["gpus"]; m=s["micro_batch"]; print(n,m,32//(n*m))')
python -c "import torch; assert torch.cuda.device_count()==$N; print('PREFLIGHT_OK',torch.__version__,torch.cuda.device_count(),flush=True)"
RUN=/workspace/ssd/codex_remote_ops/flux_piperx_20261008/runs/$TASK
mkdir -p "$RUN"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 30 > "$RUN/gpu.csv" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
python -m torch.distributed.run --standalone --nproc_per_node "$N" -m flux_action.cli train --config "$B/configs/$TASK.json" --override "shard_size=$N" --override "windows_per_rank=$MICRO" --override "grad_accumulation=$ACCUM" 2>&1 | tee -a "$RUN/train.log"
test -f "$RUN/step-3000/COMPLETE"
python -m flux_action.cli export-checkpoint --checkpoint "$RUN/step-3000" --output "$RUN/export-3000-ema" --profile ema_0p10 --dtype bfloat16
echo TRAIN_AND_EXPORT_COMPLETE > "$RUN/TRAIN_COMPLETE"
