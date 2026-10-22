#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
cd "$B"
export PATH="$B/flux-action/.venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$B/scripts" OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/test_buffered_ema.py > status/buffered-ema-unit.log 2>&1
python - <<'PY'
import pathlib,json
B=pathlib.Path.cwd();S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));R=S/'runs/two-gpu-perf-repair-v1';assert not R.exists()
c=json.loads((B/'configs/two-gpu-feasibility.json').read_text());c.update(output_dir=str(R),steps=20,checkpoint_every=20,resume='none',log_every=1)
(B/'configs/two-gpu-perf-repair.json').write_text(json.dumps(c,indent=2))
PY
R="$S/runs/two-gpu-perf-repair-v1"
mkdir -p "$R"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 1 > "$R/gpu.csv" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train_perf_worker.py configs/two-gpu-perf-repair.json > "$R/train.log" 2>&1
PERF_BUFFERED=1 python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train_perf_worker.py configs/two-gpu-perf-repair.json steps=21 resume=auto > "$R/resume.log" 2>&1
test -s "$R/step-21/COMPLETE"
python -m flux_action.cli export-checkpoint --checkpoint "$R/step-21" --output "$R/export-21-ema" --profile ema_0p10 --dtype bfloat16 > "$R/export.log" 2>&1
python - <<'PY'
import pathlib,json,hashlib,time,statistics,math
B=pathlib.Path.cwd();R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/'runs/two-gpu-perf-repair-v1'
m=json.loads((R/'export-21-ema/manifest.json').read_text())
for n,h in m['sha256'].items():
 with (R/'export-21-ema'/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
rows=[json.loads(x) for x in (R/'metrics.jsonl').read_text().splitlines()];assert any(r.get('event')=='resume' and r['step']==20 for r in rows)
assert all(math.isfinite(r[k]) for r in rows if 'loss' in r for k in ['loss','grad_norm'])
phases=[json.loads(x) for x in (R/'phases-rank0.jsonl').read_text().splitlines()]
summary={}
for mode,lo,hi in [('baseline',2,5),('buffered',8,20)]:
 x=[r for r in phases if lo<=r['step']<=hi];summary[mode]={'median_update_seconds':statistics.median(r['update_seconds'] for r in x),'phases_median':{k:statistics.median(r['phases'][k] for r in x) for k in x[0]['phases']},'compute_other_median':statistics.median(r['compute_and_other_seconds'] for r in x)}
record={'time':time.time(),'save_resume_export':True,'candidate_valid':True,'summary':summary,'speedup':summary['baseline']['median_update_seconds']/summary['buffered']['median_update_seconds'],'run':str(R),'production_restart_authorized_by_gate':False}
(B/'status/two-gpu-perf-repair-result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
PY
