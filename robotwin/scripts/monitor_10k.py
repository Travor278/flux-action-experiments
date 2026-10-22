import json,time,pathlib,statistics,csv
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 R=S/f'runs/{task}-10k-2gpu';a={'exists':R.exists()};rows=[]
 if (R/'metrics.jsonl').exists():
  for line in (R/'metrics.jsonl').read_text().splitlines():
   try: rows.append(json.loads(line))
   except json.JSONDecodeError: pass
  steps=[r for r in rows if 'step_time' in r];a['last_update']=steps[-1] if steps else None;a['last_events']=rows[-3:]
  steady=[r for r in steps[-30:] if r['step']>220]
  intervals=[(y['time']-x['time'])/(y['step']-x['step']) for x,y in zip(steady,steady[1:]) if 0<y['step']-x['step']<=10 and x['step']%1000!=0]
  if intervals:
   a['seconds_per_update']=statistics.median(intervals);a['eta_seconds']=(10000-steady[-1]['step'])*a['seconds_per_update']
 if (R/'train.log').exists():
  text=(R/'train.log').read_text(errors='replace');a['ranks']=sorted({json.loads(line)['rank'] for line in text.splitlines() if line.startswith('{') and 'TWO_GPU_RANK_READY' in line and line.endswith('}')});a['tail']=text[-1200:]
  if 'Traceback' in text:a['error']=text[text.index('Traceback'):][:5000]
 if (R/'gpu.csv').exists():
  a['gpu_recent']=list(csv.reader((R/'gpu.csv').read_text().splitlines()))[-4:]
 a['complete']=(R/'TRAIN_COMPLETE.json').exists();a['checkpoints']=[{'step':p.name,'complete':(p/'COMPLETE').exists()} for p in R.glob('step-*')];a['exports']=[p.name for p in R.glob('export-*-verified.json')]
 out['tasks'][task]=a
(B/'status/ten-k-monitor.json').write_text(json.dumps(out,indent=2))
