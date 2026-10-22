#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
TASK=${1:?task required}
case "$TASK" in click_bell|handover_block) ;; *) exit 2;; esac
cd "$B"
export PATH="$B/flux-action/.venv/bin:$PATH" PYTHONPATH="$B/flux-action-tail-v1/src:$B/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CPU_EMA_MODE=async CPU_THREADS=4 OMP_NUM_THREADS=4
unset BENCH_NO_SAVE
exec 9>"$B/status/tail-v1-$TASK.lock"
flock -n 9 || exit 3
python - <<'PY'
import pathlib,json,hashlib,torch
B=pathlib.Path.cwd();r=json.loads((B/'status/tail-repair-ready.json').read_text())
assert r['complete'] and r['data_verified'] and r['tests_passed']
for rel,h in r['files'].items():assert hashlib.sha256((B/rel).read_bytes()).hexdigest()==h,rel
assert torch.cuda.device_count()==2 and all('H100' in torch.cuda.get_device_name(i) for i in range(2))
print('TAIL_REPAIR_PREFLIGHT_PASS',r['test_summary'],flush=True)
PY
RUN="$S/runs/$TASK-10k-tail-v1"
mkdir -p "$RUN"
if test -e "$RUN/STOP_AFTER_UPDATE" || test -s "$RUN/STOPPED_SAFELY.json"; then exit 0;fi
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 5 >> "$RUN/gpu.csv" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
# First complete save at 4, exact restore and update to 5, then early100 and native keep1.
for LIMIT in 4 5 100 10000; do
 if test -e "$RUN/STOP_AFTER_UPDATE" || test -s "$RUN/STOPPED_SAFELY.json"; then exit 0;fi
 if test -s "$RUN/stage-$LIMIT-verified.json"; then continue;fi
 python -m torch.distributed.run --standalone --nproc_per_node=2 "$B/scripts/train_tail_worker.py" "$B/configs/$TASK-10k-tail-v1.json" "steps=$LIMIT" resume=auto 2>&1 | tee -a "$RUN/train.log"
 if test -s "$RUN/STOPPED_SAFELY.json"; then exit 0;fi
 python "$B/scripts/verify_tail_stage.py" "$TASK" "$LIMIT"
done
python -m flux_action.cli export-checkpoint --checkpoint "$RUN/step-10000" --output "$RUN/export-10000-ema" --profile ema_0p10 --dtype bfloat16 >> "$RUN/export.log" 2>&1
python "$B/scripts/verify_tail_stage.py" "$TASK" 10000 --export
