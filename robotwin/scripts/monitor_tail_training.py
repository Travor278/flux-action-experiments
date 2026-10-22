import json,time,pathlib,statistics,csv
from tail_log_records import log_dicts
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 R=S/f'runs/{task}-10k-tail-v1';a={'exists':R.exists()};rows=[]
 if (R/'metrics.jsonl').exists():
  for line in (R/'metrics.jsonl').read_text().splitlines():
   try:
    record=json.loads(line)
    if isinstance(record,dict):rows.append(record)
   except json.JSONDecodeError: pass
  steps=[r for r in rows if 'step_time' in r];a['last_update']=steps[-1] if steps else None;a['last_events']=rows[-3:]
  steady=[r for r in steps[-30:] if r['step']>220]
  intervals=[(y['time']-x['time'])/(y['step']-x['step']) for x,y in zip(steady,steady[1:]) if 0<y['step']-x['step']<=10 and x['step']%1000!=0]
  if intervals:
   a['seconds_per_update']=statistics.median(intervals);a['eta_seconds']=(10000-steady[-1]['step'])*a['seconds_per_update']
 if (R/'train.log').exists():
  text=(R/'train.log').read_text(errors='replace');a['ranks']=sorted({r['rank'] for r in log_dicts(text) if r.get('event')=='EFFICIENCY_RANK_READY'});a['tail']=text[-1200:]
  if 'Traceback' in text:a['error']=text[text.index('Traceback'):][:5000]
 if (R/'gpu.csv').exists():
  a['gpu_recent']=list(csv.reader((R/'gpu.csv').read_text().splitlines()))[-4:]
 a['complete']=(R/'TRAIN_COMPLETE.json').exists();a['checkpoints']=[{'step':p.name,'complete':(p/'COMPLETE').exists()} for p in R.glob('step-*')];a['exports']=[p.name for p in R.glob('export-*-verified.json')]
 for label,name in [('early_checkpoint','stage-100-verified.json'),('first_save','stage-4-verified.json'),('exact_resume_gate','stage-5-verified.json'),('safe_stop','STOPPED_SAFELY.json')]:
  if (R/name).exists():a[label]=json.loads((R/name).read_text())
 a['gpu_tail_preflight']=[r for r in log_dicts((R/'train.log').read_text(errors='replace')) if r.get('event')=='TAIL_GPU_PREFLIGHT_PASS'] if (R/'train.log').exists() else []
 a['resume_events']=[r for r in rows if r.get('event')=='resume'][-3:]
 out['tasks'][task]=a
(B/'status/tail-training-monitor.json').write_text(json.dumps(out,indent=2))
