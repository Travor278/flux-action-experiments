from pathlib import Path
import json,time,hashlib
from eval150_contract import TASKS,SEEDS,output,aggregate
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
plan_sha=hashlib.sha256((R/'status/r2-eval150-plan.json').read_bytes()).hexdigest();tasks={}
for task in TASKS:
 rows=[]
 for seed in SEEDS:
  out=output(R,task,seed);r=json.loads((out/'COMPLETE.json').read_text())
  assert r['task']==task and r['eval_seed']==seed and r['plan_sha256']==plan_sha
  assert hashlib.sha256((out/'result.json').read_bytes()).hexdigest()==r['result_sha256']
  rows.append({k:v for k,v in r.items() if k!='videos'})
 tasks[task]={'aggregate':aggregate(rows),'runs':rows}
r={'time':time.time(),'verified':True,'tasks':tasks,'total_episodes':300,'total_videos':900,'training_seed':42,'eval_seeds':list(SEEDS),'official_verified_listing':False,'note':'Local reproducible protocol results, not an endorsed cloud leaderboard submission'}
(R/'status/r2-eval150-complete.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
