R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'));run=S/'runs/perf-exact-20261014';out=S/'runs/match_and_pick_from_conveyor-10k-prefix-v1'
production=None
for line in (out/'metrics.jsonl').read_text().splitlines():
 try:r=json.loads(line)
 except ValueError:continue
 if isinstance(r,dict) and r.get('step')==720 and 'step_time' in r:production=r
assert production
baseline=json.loads((run/'baseline/comparison-rank0.json').read_text())['updates'][0]['metrics'];cached=json.loads((run/'cached/comparison-rank0.json').read_text())['updates'][0]['metrics']
keys=['loss','grad_norm','video_mse','action_mse','lr_trunk','lr_heads']
r={'time':time.time(),'step':720,'production_without_cache':{k:production[k] for k in keys},'diagnostic_without_cache':{k:baseline[k] for k in keys},'diagnostic_cached':{k:cached[k] for k in keys},'scope':'Same original719 checkpoint and exact resume. Production vs diagnostic instrumentation differs; useful repeat evidence, not a complete baseline-baseline final-state controlled experiment.'}
(R/'status/first-update-repeat-evidence.json').write_text(json.dumps(r,indent=2))
PY
