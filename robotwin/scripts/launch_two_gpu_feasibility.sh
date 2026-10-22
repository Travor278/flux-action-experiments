#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
cd "$B"
export PATH="$B/flux-action/.venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$B/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c 'import torch;assert torch.cuda.device_count()==2;assert all("H100" in torch.cuda.get_device_name(i) for i in range(2))'
EMA_TEST_CUDA=1 python -m torch.distributed.run --standalone --nproc_per_node=2 "$B/scripts/test_cpu_power_ema.py" > "$B/status/cpu-ema-gpu-test.log" 2>&1
python - <<'PY'
import pathlib,json
B=pathlib.Path.cwd();S=pathlib.Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
run=S/'runs/two-gpu-feasibility-v1';assert not run.exists(),'Preserve previous diagnostic run'
cfg=json.loads((B/'configs/handover_block.json').read_text());cfg.update(output_dir=str(run),steps=12,frozen_steps=0,trunk_warmup_steps=1,heads_warmup_steps=1,cooldown_start=None,cooldown_steps=0,shard_size=2,windows_per_rank=1,grad_accumulation=16,checkpoint_every=12,log_every=1,resume='none')
(B/'configs/two-gpu-feasibility.json').write_text(json.dumps(cfg,indent=2))
PY
RUN="$S/runs/two-gpu-feasibility-v1"
mkdir -p "$RUN"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 1 > "$RUN/gpu.csv" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
python -m torch.distributed.run --standalone --nproc_per_node=2 "$B/scripts/train_two_gpu_worker.py" "$B/configs/two-gpu-feasibility.json" 2>&1 | tee "$RUN/train.log"
test -s "$RUN/step-12/COMPLETE"
python -m torch.distributed.run --standalone --nproc_per_node=2 "$B/scripts/train_two_gpu_worker.py" "$B/configs/two-gpu-feasibility.json" steps=13 resume=auto 2>&1 | tee "$RUN/resume.log"
test -s "$RUN/step-13/COMPLETE"
python -m flux_action.cli export-checkpoint --checkpoint "$RUN/step-13" --output "$RUN/export-13-ema" --profile ema_0p10 --dtype bfloat16 > "$RUN/export.log" 2>&1
python - <<'PY'
import json,pathlib,statistics,math,time,hashlib
B=pathlib.Path.cwd();R=pathlib.Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008/runs/two-gpu-feasibility-v1')
rows=[json.loads(x) for x in (R/'metrics.jsonl').read_text().splitlines()]
steps=[x for x in rows if 'step_time' in x];assert len(steps)==13 and all(math.isfinite(x[k]) for x in steps for k in ['loss','grad_norm','step_time'])
assert any(x.get('event')=='resume' and x['step']==12 for x in rows)
steady=[x for x in steps if 5<=x['step']<=12];rate=statistics.median(b['time']-a['time'] for a,b in zip(steady,steady[1:]))
manifest=json.loads((R/'export-13-ema/manifest.json').read_text())
for n,h in manifest['sha256'].items():
 with (R/'export-13-ema'/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
state=json.loads((R/'step-13/state.json').read_text());assert state['step']==13 and state['world_size']==2
record={'time':time.time(),'verified':True,'world_size':2,'micro_batch':1,'gradient_accumulation':16,'global_batch':32,'param_dtype':'float32','ema_dtype':'float32','ema_device':'cpu','sync_each_microbatch':True,'save_resume_export':True,'seconds_per_update':rate,'windows_per_second':32/rate,'peak_gpu_gb':max(x['peak_mem_gb'] for x in steady),'estimated_10000_steps_seconds':10000*rate,'run':str(R)}
(B/'status/two-gpu-feasibility-verified.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
PY
