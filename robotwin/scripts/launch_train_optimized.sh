#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
TASK=${1:?task required}
case "$TASK" in click_bell|handover_block) ;; *) exit 2;; esac
cd "$B"
export PATH="$B/flux-action/.venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$B/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BENCH_NO_SAVE
read -r CPU_EMA_MODE CPU_THREADS < <(python -c 'import json; r=json.load(open("status/optimized-ten-k-ready.json"));print(r["selected"]["ema_mode"],r["selected"]["cpu_threads"])')
export CPU_EMA_MODE CPU_THREADS OMP_NUM_THREADS="$CPU_THREADS"
exec 9>"$B/status/optimized-ten-k-$TASK.lock"
flock -n 9 || exit 3
python - <<'PY'
import pathlib,json,hashlib,torch
B=pathlib.Path.cwd();r=json.loads((B/'status/optimized-ten-k-ready.json').read_text());assert r['verified']
p=json.loads((B/'status/efficiency-selected-verified.json').read_text());assert all(p[k] for k in ['complete','save_resume_export','safe_stop_verified'])
for rel,h in r['files'].items():assert hashlib.sha256((B/rel).read_bytes()).hexdigest()==h,rel
assert torch.cuda.device_count()==2 and all('H100' in torch.cuda.get_device_name(i) for i in range(2))
print('OPTIMIZED_TEN_K_PREFLIGHT_PASS',r['selected'],flush=True)
PY
RUN="$S/runs/$TASK-10k-opt-v1"
mkdir -p "$RUN"
if test -e "$RUN/STOP_AFTER_UPDATE" || test -s "$RUN/STOPPED_SAFELY.json"; then echo 'STOP_REQUEST_PRESERVED';exit 0;fi
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 5 >> "$RUN/gpu.csv" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
for LIMIT in 100 1000 3000 5000 10000; do
 if test -e "$RUN/STOP_AFTER_UPDATE" || test -s "$RUN/STOPPED_SAFELY.json"; then echo 'STOP_REQUEST_PRESERVED';exit 0;fi
 if [ "$LIMIT" = 100 ] && test -s "$RUN/EARLY_CHECKPOINT_VERIFIED.json"; then continue;fi
 if [ "$LIMIT" != 100 ] && test -s "$RUN/export-$LIMIT-verified.json"; then continue;fi
 python -m torch.distributed.run --standalone --nproc_per_node=2 "$B/scripts/train_efficiency_worker.py" "$B/configs/$TASK-10k-opt-v1.json" "steps=$LIMIT" resume=auto 2>&1 | tee -a "$RUN/train.log"
 if test -s "$RUN/STOPPED_SAFELY.json"; then echo 'STOPPED_WITH_COMPLETE_CHECKPOINT';exit 0;fi
 test -s "$RUN/step-$LIMIT/COMPLETE"
 if [ "$LIMIT" = 100 ]; then
  python -c 'import pathlib,json,sys,time;p=pathlib.Path(sys.argv[1]);r=json.loads((p/"step-100/state.json").read_text());assert r["step"]==100 and r["world_size"]==2;(p/"EARLY_CHECKPOINT_VERIFIED.json").write_text(json.dumps({"step":100,"time":time.time(),"verified":True}))' "$RUN"
  continue
 fi
 python -m flux_action.cli export-checkpoint --checkpoint "$RUN/step-$LIMIT" --output "$RUN/export-$LIMIT-ema" --profile ema_0p10 --dtype bfloat16 >> "$RUN/export.log" 2>&1
 python "$B/scripts/verify_optimized_export.py" "$TASK" "$LIMIT" | tee -a "$RUN/export.log"
done
