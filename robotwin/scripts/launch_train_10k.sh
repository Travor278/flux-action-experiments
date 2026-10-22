#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
TASK=${1:?task required}
case "$TASK" in click_bell|handover_block) ;; *) exit 2;; esac
cd "$B"
export PATH="$B/flux-action/.venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$B/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLUX_10K_TASK="$TASK"
exec 9>"$B/status/ten-k-$TASK.lock"
flock -n 9 || { echo 'Task already active'; exit 3; }
python - <<'PY'
import json,pathlib,hashlib,torch,os
B=pathlib.Path.cwd();r=json.loads((B/'status/ten-k-ready.json').read_text())
assert r['verified'] and r['recipe']['save_resume_export']
for path,sha in r['files'].items(): assert hashlib.sha256((B/path).read_bytes()).hexdigest()==sha, path
assert torch.cuda.device_count()==2 and all('H100' in torch.cuda.get_device_name(i) for i in range(2))
print('TEN_K_PREFLIGHT_OK',os.environ['FLUX_10K_TASK'],torch.__version__,flush=True)
PY
RUN="$S/runs/$TASK-10k-2gpu"
mkdir -p "$RUN"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 30 >> "$RUN/gpu.csv" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
# Only stop for milestone exports. LR schedule remains fixed at warmup200/600 and cooldown9000+1000;
# stock resume restores optimizer, both CPU EMA profiles, RNG, sampler and absolute scheduler step.
for LIMIT in 1000 3000 5000 10000; do
  if test -s "$RUN/export-$LIMIT-verified.json"; then continue; fi
  python -m torch.distributed.run --standalone --nproc_per_node=2 "$B/scripts/train_two_gpu_worker.py" "$B/configs/$TASK-10k-2gpu.json" "steps=$LIMIT" resume=auto 2>&1 | tee -a "$RUN/train.log"
  test -s "$RUN/step-$LIMIT/COMPLETE"
  python -m flux_action.cli export-checkpoint --checkpoint "$RUN/step-$LIMIT" --output "$RUN/export-$LIMIT-ema" --profile ema_0p10 --dtype bfloat16 >> "$RUN/export.log" 2>&1
  python "$B/scripts/verify_10k_export.py" "$TASK" "$LIMIT" | tee -a "$RUN/export.log"
done
