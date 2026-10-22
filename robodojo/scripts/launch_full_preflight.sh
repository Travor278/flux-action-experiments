#!/usr/bin/env bash
set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
S=${R/\/hdd\//\/ssd\/}
OUT="$S/runs/full-preflight-v1"
export PATH="$R/policy-venv/bin:$PATH" PYTHONPATH="$R/scripts:$R/flux-action/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BENCH_NO_SAVE
exec 9>"$R/status/full-preflight.lock"
flock -n 9 || exit 3
test ! -e "$OUT"
mkdir -p "$OUT"
exec > >(tee -a "$R/status/full-preflight-launch.log") 2>&1
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 2 > "$OUT/gpu.csv" &
mon=$!
trap 'kill "$mon" 2>/dev/null || true' EXIT
python -m torch.distributed.run --standalone --nproc_per_node=2 "$R/scripts/train_full_preflight.py" "$R/configs/full-preflight-v1.json" > "$OUT/train.log" 2>&1
test -f "$OUT/step-2/COMPLETE"
python -m torch.distributed.run --standalone --nproc_per_node=2 "$R/scripts/train_full_preflight.py" "$R/configs/full-preflight-v1.json" steps=3 resume=auto > "$OUT/resume.log" 2>&1
test -f "$OUT/step-3/COMPLETE"
test -f "$OUT/exact-resume-rank0.json"
test -f "$OUT/exact-resume-rank1.json"
python -m flux_action.cli export-checkpoint --checkpoint "$OUT/step-3" --output "$OUT/export-3-ema" --profile ema_0p10 --dtype bfloat16 > "$OUT/export.log" 2>&1
python - <<'PY'
from pathlib import Path
import hashlib,json,math,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1'
rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines() if l.strip()]
assert any(r.get('event')=='resume' and r.get('step')==2 for r in rows)
updates=[r for r in rows if 'step_time' in r];assert [r['step'] for r in updates]==[1,2,3]
assert all(math.isfinite(r[k]) for r in updates for k in ['loss','grad_norm'])
export=out/'export-3-ema';m=json.loads((export/'manifest.json').read_text())
for n,h in m['sha256'].items():
 with (export/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
assert json.loads((export/'config.json').read_text())['prefix_memory_spec']['version']=='causal_prefix_rgb_v1'
(R/'status/full-preflight-complete.json').write_text(json.dumps({'time':time.time(),'complete':True,'steps':[1,2,3],'resume_from':2,'export':str(export),'scope':'Full model diagnostic save/resume/export; separate artifact/parameter/RNG audit and exported inference still required'},indent=2))
PY
