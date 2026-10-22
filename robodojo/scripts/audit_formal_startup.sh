R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/formal-startup-audit.log" 2>&1
from pathlib import Path
import json,time,csv,statistics
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'));result={'time':time.time(),'tasks':{}}
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 out=S/'runs'/f'{task}-10k-prefix-v1';text=(out/'early.log').read_text(errors='replace');ready=[]
 for line in text.splitlines():
  if 'ROBODOJO_FORMAL_RANK_READY' in line:
   i=line.index('{');ready.append(json.loads(line[i:]))
 assert sorted(r['rank'] for r in ready)==[0,1],ready
 assert all(r['global_batch']==32 for r in ready)
 rows=list(csv.reader((out/'gpu.csv').read_text().splitlines()))[-300:];gpus={}
 for row in rows:
  if len(row)!=4 or not row[1].strip().isdigit():continue
  key=row[1].strip();gpus.setdefault(key,[]).append((float(row[2].strip().split()[0]),int(row[3].strip().split()[0])))
 result['tasks'][task]={'ranks':ready,'gpu_recent_mean':{k:statistics.mean(v[0] for v in xs) for k,xs in gpus.items()},'gpu_recent_max_mib':{k:max(v[1] for v in xs) for k,xs in gpus.items()},'scope':'Actual two-rank startup and recent300telemetry lines; first200updates frozen trunkLR0, notfullrunETA.'}
(R/'status/formal-startup-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
PY
