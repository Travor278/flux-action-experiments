#!/usr/bin/env bash
set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
S=${R/\/hdd\//\/ssd\/}
OUT="$S/runs/full-preflight-v1"
export PATH="$R/policy-venv/bin:$PATH" PYTHONPATH="$R/scripts:$R/flux-action/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BENCH_NO_SAVE
exec 9>"$R/status/full-preflight.lock"
flock -n 9 || exit 3
test -f "$OUT/step-2/COMPLETE"
test ! -e "$OUT/step-3/COMPLETE"
test ! -e "$OUT/STOP_AFTER_UPDATE"
exec > >(tee -a "$R/status/full-preflight-recovery-launch.log") 2>&1
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 2 > "$OUT/gpu-recovery.csv" &
mon=$!
trap 'kill "$mon" 2>/dev/null || true' EXIT
RECOVER_UNFINGERPRINTED_STEP2=1 python -m torch.distributed.run --standalone --nproc_per_node=2 "$R/scripts/train_full_preflight.py" "$R/configs/full-preflight-v1.json" steps=3 resume=auto > "$OUT/resume2.log" 2>&1
test -f "$OUT/step-3/COMPLETE"
test -f "$OUT/exact-state-step3-rank0.json"
test -f "$OUT/exact-state-step3-rank1.json"
python -m torch.distributed.run --standalone --nproc_per_node=2 "$R/scripts/train_full_preflight.py" "$R/configs/full-preflight-v1.json" steps=4 resume=auto > "$OUT/resume3.log" 2>&1
test -f "$OUT/step-4/COMPLETE"
test -f "$OUT/exact-resume-step3-rank0.json"
test -f "$OUT/exact-resume-step3-rank1.json"
python -m flux_action.cli export-checkpoint --checkpoint "$OUT/step-4" --output "$OUT/export-4-ema" --profile ema_0p10 --dtype bfloat16 > "$OUT/export.log" 2>&1
python - <<'PY'
from pathlib import Path
import hashlib,json,math,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1'
rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines() if l.strip()]
for step in (2,3):assert any(r.get('event')=='resume' and r.get('step')==step for r in rows)
updates=[r for r in rows if 'step_time' in r];assert [r['step'] for r in updates]==[1,2,3,4]
assert all(math.isfinite(r[k]) for r in updates for k in ['loss','grad_norm'])
for rank in (0,1):
 p=json.loads((out/f'exact-resume-step3-rank{rank}.json').read_text());assert p['verified'] and p['step']==3
export=out/'export-4-ema';m=json.loads((export/'manifest.json').read_text())
for n,h in m['sha256'].items():
 with (export/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
assert json.loads((export/'config.json').read_text())['prefix_memory_spec']['version']=='causal_prefix_rgb_v1'
(R/'status/full-preflight-complete.json').write_text(json.dumps({'time':time.time(),'complete':True,'steps':[1,2,3,4],'exact_fingerprint_restore_step':3,'step2_presave_fingerprint_unavailable':True,'export':str(export),'scope':'Full diagnostic; saved step2 reused, exact before/after state hashes proven at3; separate artifact and inference audits still required'},indent=2))
PY
