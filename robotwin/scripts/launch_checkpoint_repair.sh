#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
cd "$B"
export PATH="$B/flux-action/.venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$B/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -c 'import torch;assert torch.cuda.device_count()==8;assert all("H100" in torch.cuda.get_device_name(i) for i in range(8));print("CHECKPOINT_REPAIR_8H100_PREFLIGHT",flush=True)'
python - <<'PY'
import json,pathlib
B=pathlib.Path.cwd();S=pathlib.Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
r=S/'runs/checkpoint-repair-v1';assert not r.exists(),'Preserve prior smoke result'
c=json.loads((B/'configs/handover_block.json').read_text());c.update(output_dir=str(r),steps=2,frozen_steps=0,trunk_warmup_steps=1,heads_warmup_steps=1,cooldown_start=None,cooldown_steps=0,shard_size=8,windows_per_rank=2,grad_accumulation=2,checkpoint_every=2,keep_checkpoints=1,log_every=1,resume='none');(B/'configs/checkpoint-repair.json').write_text(json.dumps(c,indent=2))
PY
RUN="$S/runs/checkpoint-repair-v1"
mkdir -p "$RUN"
python -m torch.distributed.run --standalone --nproc_per_node=8 -m flux_action.cli train --config "$B/configs/checkpoint-repair.json" 2>&1 | tee "$B/status/checkpoint-repair-save.log"
test -s "$RUN/step-2/COMPLETE"
python -m torch.distributed.run --standalone --nproc_per_node=8 -m flux_action.cli train --config "$B/configs/checkpoint-repair.json" --override steps=3 --override resume=auto 2>&1 | tee "$B/status/checkpoint-repair-resume.log"
test -s "$RUN/step-3/COMPLETE"
python -m flux_action.cli export-checkpoint --checkpoint "$RUN/step-3" --output "$RUN/export-3-ema" --profile ema_0p10 --dtype bfloat16 2>&1 | tee "$B/status/checkpoint-repair-export.log"
python - <<'PY'
import json,pathlib,time
B=pathlib.Path.cwd();r=pathlib.Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008/runs/checkpoint-repair-v1');c=r/'step-3'
state=json.loads((c/'state.json').read_text());assert state['step']==3 and state['world_size']==8
for name in ['model','optimizer','ema_0p10','ema_0p05']:
 assert (c/name/'.metadata').is_file();assert len(list((c/name).glob('*.distcp')))>0
assert len(list((c/'rng').glob('rank-*.pt')))==8
from safetensors import safe_open
files=list((r/'export-3-ema').rglob('*.safetensors'));assert files
for f in files:
 with safe_open(f,framework='pt',device='cpu') as s:assert len(s.keys())>0
(B/'status/checkpoint-repair-verified.json').write_text(json.dumps({'time':time.time(),'step':3,'world_size':8,'checkpoint':str(c),'export':str(r/'export-3-ema'),'model_optimizer_dual_ema':True,'save_resume_export':True},indent=2))
PY
