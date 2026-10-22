set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,hashlib,math
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');run=Path(str(R).replace('/hdd/','/ssd/'))/'runs/perf-exact-20261014'
result=json.loads((run/'result.json').read_text());assert result['exit_code']==0 and not result['verified'] and not result['production_optimization_adopted']
proof=[]
for rank in (0,1):
 a=json.loads((run/f'baseline/comparison-rank{rank}.json').read_text());b=json.loads((run/f'cached/comparison-rank{rank}.json').read_text())
 assert a['start']==b['start']==719 and a['end']==b['end']==727
 assert a['before']==b['before']
 assert a['after']['rng']==b['after']['rng'] and a['after']['data_position']==b['after']['data_position']
 assert all(math.isfinite(u['metrics'][k]) for x in (a,b) for u in x['updates'] for k in ('loss','grad_norm'))
 proof.append({'rank':rank,'before_exact':True,'after_rng_and_data_exact':True,'after_equal_components':{k:a['after'][k]==b['after'][k] for k in a['after']},'first_loss_equal':a['updates'][0]['metrics']['loss']==b['updates'][0]['metrics']['loss'],'records_sha256':{mode:hashlib.sha256((run/f'{mode}/comparison-rank{rank}.json').read_bytes()).hexdigest() for mode in ['baseline','cached']}})
formal=json.loads((R/'status/formal-ready.json').read_text())
for name,h in formal['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
out=Path(str(R).replace('/hdd/','/ssd/'))/'runs/match_and_pick_from_conveyor-10k-prefix-v1'
assert not (out/'STOP_AFTER_UPDATE').exists() and not (out/'STOPPED_SAFELY.json').exists()
rows=[]
for line in (out/'metrics.jsonl').read_text().splitlines():
 try:x=json.loads(line)
 except ValueError:continue
 if isinstance(x,dict):rows.append(x)
assert any(x.get('event')=='resume' and x.get('step')==719 and x.get('mode')=='exact' for x in rows)
assert any(x.get('step',0)>727 and 'step_time' in x for x in rows)
r={'time':time.time(),'audit_complete':True,'optimization_adopted':False,'all19_production_hashes_unchanged':True,'source719_production_exact_resume_and_fresh_updates':True,'ranks':proof,'scope':'Full-update bitwise criterion failed; do not attribute differences solely to caching without baseline repeat controls. No optimizer/model diagnostic state copied into production.'}
(R/'status/perf-exact-independent-audit.json').write_text(json.dumps(r,indent=2))
PY
