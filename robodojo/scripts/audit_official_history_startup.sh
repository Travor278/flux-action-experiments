set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/official-history-startup-audit.log" 2>&1
from pathlib import Path
import json,time,csv,statistics,hashlib,datetime,shutil
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
g=json.loads((R/'status/official-history-ready.json').read_text())
for n,h in g['files'].items():
 with (R/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,n
result={'time':time.time(),'verified':True,'official_source_config_sha_unchanged':True,'tasks':{},'free_ssd_bytes':shutil.disk_usage(S).free}
dec=json.JSONDecoder()
def objects(text):
 result=[]
 for line in text.splitlines():
  offset=0
  while True:
   i=line.find('{',offset)
   if i<0:break
   try:r,n=dec.raw_decode(line[i:])
   except json.JSONDecodeError:offset=i+1;continue
   if isinstance(r,dict):result.append(r)
   offset=i+n
 return result
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 out=S/'runs'/f'{task}-10k-official-history-v1'
 cfg=json.loads((R/f'configs/{task}-10k-official-history-v1.json').read_text());assert cfg['dataset']=='index' and 'prefix_memory_spec' not in cfg['policy']
 text=(out/'early.log').read_text(errors='replace');ready=[r for r in objects(text) if r.get('event')=='ROBODOJO_FORMAL_RANK_READY'];assert sorted(r['rank'] for r in ready)==[0,1]
 records=objects((out/'metrics.jsonl').read_text());updates=[r for r in records if 'step_time' in r]
 assert all(__import__('math').isfinite(r[k]) for r in updates for k in ('loss','grad_norm'))
 initial=json.loads((out/'INITIAL_CHECKPOINT_VERIFIED.json').read_text());early=json.loads((out/'EARLY_CHECKPOINT_VERIFIED.json').read_text());assert initial['verified'] and early['verified']
 assert any(r.get('event')=='resume' and r['step']==2 and r['mode']=='exact' for r in records)
 measured=[r for r in updates if 30<=r['step']<=90];lo=measured[0]['time'];hi=measured[-1]['time'];gpus={}
 for row in csv.reader((out/'gpu.csv').read_text().splitlines()):
  if len(row)!=4 or not row[1].strip().isdigit():continue
  t=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
  if lo<=t<=hi:gpus.setdefault(row[1].strip(),[]).append(float(row[2].strip().split()[0]))
 assert len(gpus)==2 and all(len(v)>100 for v in gpus.values())
 errors={n:[line for line in (out/n).read_text(errors='replace').splitlines() if 'Traceback (most recent call last)' in line or 'CUDA out of memory' in line] for n in ['initial.log','early.log','train.log'] if (out/n).exists()}
 assert not any(errors.values()),errors
 result['tasks'][task]={'ranks':ready,'latest':updates[-1],'checkpoints':[p.parent.name for p in out.glob('step-*/COMPLETE')],'initial2_verified':True,'early100_verified':True,'resume_events':[r for r in records if r.get('event')=='resume'],'gpu_training_mean':{k:statistics.mean(v) for k,v in gpus.items()},'gpu_training_samples_per_card':{k:len(v) for k,v in gpus.items()},'gpu_interval':'Only actual30to90updates; frozen trunk, not fullrun speed','errors':errors}
result['unreferenced_transfer_candidates']=[str(p) for root in [R,S] for p in root.glob('*.incomplete')]
(R/'status/official-history-startup-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
PY
"$R/policy-venv/bin/python" "$R/scripts/monitor_official_history.py" > "$R/status/official-history-monitor-refresh.log" 2>&1
