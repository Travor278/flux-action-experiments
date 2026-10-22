#!/usr/bin/env bash
set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
S=${R/\/hdd\//\/ssd\/}
TASK=$1
case "$TASK" in cover_blocks|match_and_pick_from_conveyor);; *) exit 2;; esac
OUT="$S/runs/$TASK-10k-official-history-v1"
export PATH="$R/policy-venv/bin:$PATH" PYTHONPATH="$R/scripts:$R/flux-action-official-v1/src" PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BENCH_NO_SAVE
exec 9>"$R/status/official-history-$TASK.lock"
flock -n 9 || exit 3
python "$R/scripts/check_official_history_gate.py" "$TASK"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/launch.log") 2>&1
test ! -f "$OUT/STOPPED_SAFELY.json" || exit 0
test ! -f "$OUT/TRAIN_COMPLETE.json" || exit 0
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 2 >> "$OUT/gpu.csv" &
mon=$!
trap 'kill "$mon" 2>/dev/null || true' EXIT
if [ ! -f "$OUT/INITIAL_CHECKPOINT_VERIFIED.json" ]; then
 python -m torch.distributed.run --standalone --nproc_per_node=2 "$R/scripts/train_official_history.py" "$R/configs/$TASK-10k-official-history-v1.json" steps=2 log_every=1 >> "$OUT/initial.log" 2>&1
 test ! -f "$OUT/STOPPED_SAFELY.json" || exit 0
 python "$R/scripts/audit_official_history.py" "$TASK" 2
fi
if [ ! -f "$OUT/EARLY_CHECKPOINT_VERIFIED.json" ]; then
 python -m torch.distributed.run --standalone --nproc_per_node=2 "$R/scripts/train_official_history.py" "$R/configs/$TASK-10k-official-history-v1.json" steps=100 >> "$OUT/early.log" 2>&1
 test ! -f "$OUT/STOPPED_SAFELY.json" || exit 0
 python "$R/scripts/audit_official_history.py" "$TASK" 100
fi
python -m torch.distributed.run --standalone --nproc_per_node=2 "$R/scripts/train_official_history.py" "$R/configs/$TASK-10k-official-history-v1.json" >> "$OUT/train.log" 2>&1
test ! -f "$OUT/STOPPED_SAFELY.json" || exit 0
python -m flux_action.cli export-checkpoint --checkpoint "$OUT/step-10000" --output "$OUT/export-10000-ema" --profile ema_0p10 --dtype bfloat16 > "$OUT/export.log" 2>&1
python "$R/scripts/audit_official_history.py" "$TASK" 10000
