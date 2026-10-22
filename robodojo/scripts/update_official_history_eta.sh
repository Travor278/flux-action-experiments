set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/official-history-full-eta.log" 2>&1
from pathlib import Path
import json,time,csv,statistics,hashlib,datetime,math,shutil
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
g=json.loads((R/'status/official-history-ready.json').read_text())
for n,h in g['files'].items():
 with (R/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,n
now=time.time();result={'time':now,'official_source_config_unchanged':True,'tasks':{},'buffer_seconds_each':1800,'free_ssd_bytes':shutil.disk_usage(S).free,'temporary_candidates':[str(p) for root in [R,S] for p in root.glob('*.incomplete')]}
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 out=S/'runs'/f'{task}-10k-official-history-v1';records=[]
 for line in (out/'metrics.jsonl').read_text().splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if isinstance(r,dict):records.append(r)
 resumes=[r for r in records if r.get('event')=='resume'];assert any(r['step']==100 and r['mode']=='exact' for r in resumes)
 cutoff=max(r['time'] for r in resumes)
 restart=R/'status/official-history-restart-audit-20261015.json'
 if restart.exists():cutoff=max(cutoff,json.loads(restart.read_text())['time'])
 updates=[r for r in records if 'step_time' in r and r['time']>cutoff]
 assert all(math.isfinite(r[k]) for r in updates for k in ['loss','grad_norm'])
 stable=[r for r in updates if r['step']>=220][-31:];rates=[(b['time']-a['time'])/(b['step']-a['step']) for a,b in zip(stable,stable[1:]) if b['step']>a['step'] and a['step']//1000==b['step']//1000]
 assert len(rates)>=8
 latest=updates[-1];rate=statistics.median(rates);finish=latest['time']+(10000-latest['step'])*rate+1800
 gpus={}
 for row in csv.reader((out/'gpu.csv').read_text().splitlines()):
  if len(row)!=4 or not row[1].strip().isdigit():continue
  t=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
  if stable[0]['time']<=t<=latest['time']:gpus.setdefault(row[1].strip(),[]).append(float(row[2].strip().split()[0]))
 assert len(gpus)==2
 result['tasks'][task]={'latest':latest,'exact_resume100':True,'normal_intervals':len(rates),'seconds_per_update_median':rate,'gpu_mean':{k:statistics.mean(v) for k,v in gpus.items()},'gpu_fraction_below30':{k:sum(x<30 for x in v)/len(v) for k,v in gpus.items()},'gpu_samples_per_card':{k:len(v) for k,v in gpus.items()},'estimated_finish_epoch':finish,'remaining_seconds':max(0,finish-now),'checkpoint_complete':[p.parent.name for p in out.glob('step-*/COMPLETE')]}
late=max(r['estimated_finish_epoch'] for r in result['tasks'].values());result.update(next_check_minutes=math.ceil(max(0,late-now)/60),estimated_finish_beijing=datetime.datetime.fromtimestamp(late,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),estimate_not_guarantee=True)
(R/'status/official-history-full-eta.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
PY
