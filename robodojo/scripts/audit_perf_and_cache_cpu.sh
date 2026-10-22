#!/usr/bin/env bash
set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/scripts:$R/flux-action/src" PYTHONNOUSERSITE=1
exec > "$R/status/prefix-cache-cpu.log" 2>&1
bash -n "$R/scripts/launch_prefix_cache_probe.sh"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,ast,hashlib,statistics,csv
import torch
from exact_prefix_cache import ExactPrefixVAECache
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
class Frozen:
 def __init__(self):
  self.module=torch.nn.Linear(4,4,bias=False).requires_grad_(False);self.calls=0
 def encode_task(self,x):self.calls+=1;return self.module(x)
v=Frozen();cache=ExactPrefixVAECache(v,max_bytes=32);x=torch.arange(4,dtype=torch.float32).reshape(1,4)
a=cache.encode_task(x);b=cache.encode_task(x.clone());assert torch.equal(a,b) and v.calls==1
b.add_(99);assert torch.equal(cache.encode_task(x),a),'Returned cache must not alias caller mutation'
with torch.no_grad():v.module.weight.add_(1)
c=cache.encode_task(x);assert v.calls==2 and not torch.equal(a,c)
cache.encode_task(x+1);cache.encode_task(x+2);assert cache.bytes<=32
cache.encode_task(x);assert v.calls==5,'LRU must recompute evicted input'
class RandomVAE(Frozen):
 def encode_task(self,x):return super().encode_task(x)+torch.rand_like(x)
try:ExactPrefixVAECache(RandomVAE()).encode_task(x)
except AssertionError:pass
else:raise AssertionError('Stochastic encoding must be refused')
g=json.loads((R/'status/prefix-cache-ready.json').read_text(encoding='utf-8-sig'))
for n,d in g['files'].items():
 p=R/'scripts'/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==d
 if n.endswith('.py'):ast.parse(p.read_text())
(R/'status/prefix-cache-cpu-verified.json').write_text(json.dumps({'time':time.time(),'verified':True,'cases':['byte-identical hit','caller mutation isolation','weight version invalidation','bounded eviction','stochastic rejection'],'files':g['files']},indent=2))
result={'time':time.time(),'tasks':{},'scope':'Passive telemetry only; no production code/config/process changes'}
S=Path(str(R).replace('/hdd/','/ssd/'))
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 out=S/'runs'/f'{task}-10k-prefix-v1';rows=[]
 for line in (out/'metrics.jsonl').read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if isinstance(r,dict) and r.get('step',0)>=220 and 'step_time' in r:rows.append(r)
 rows=rows[-30:];gpu={}
 for row in list(csv.reader((out/'gpu.csv').read_text().splitlines()))[-600:]:
  try:i=int(row[1]);u=float(row[2].strip().split()[0]);m=float(row[3].strip().split()[0])
  except (ValueError,IndexError):continue
  gpu.setdefault(i,[]).append((u,m))
 result['tasks'][task]={'latest':rows[-1],'median_step_seconds':statistics.median(r['step_time'] for r in rows),'median_data_wait_seconds':statistics.median(r.get('data_wait',0) for r in rows),'gpu':{i:{'samples':len(v),'mean_utilization':statistics.mean(x[0] for x in v),'minimum':min(x[0] for x in v),'maximum':max(x[0] for x in v),'fraction_below30':sum(x[0]<30 for x in v)/len(v),'fraction_atleast90':sum(x[0]>=90 for x in v)/len(v),'peak_mib':max(x[1] for x in v)} for i,v in gpu.items()}}
(R/'status/performance-baseline-20261014.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
PY
"$R/policy-venv/bin/python" "$R/scripts/monitor_formal_training.py"
