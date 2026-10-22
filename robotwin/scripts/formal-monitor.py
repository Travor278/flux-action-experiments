from pathlib import Path
import json,time,csv,math,statistics,shutil,re
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
out={'time':time.time(),'ssd_free_bytes':shutil.disk_usage(S).free,'tasks':{}}
for task in ['click_bell','handover_block']:
 p=S/'runs'/task;r={'exists':p.exists()}
 if (p/'metrics.jsonl').exists():
  rows=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines() if x.strip()]
  steps=[x for x in rows if 'step_time' in x]
  r['last_steps']=steps[-3:];r['events']=[x for x in rows if 'event' in x][-5:]
  r['all_logged_steps_finite']=all(math.isfinite(x[k]) for x in steps for k in ['loss','grad_norm','step_time'])
  if len(steps)>1:
   rate=statistics.median((b['time']-a['time'])/(b['step']-a['step']) for a,b in zip(steps[-20:],steps[-20:][1:]) if b['step']>a['step'])
   last=steps[-1];r['seconds_per_step']=rate;r['next_checkpoint_eta_seconds']=(min(3000,(last['step']//1000+1)*1000)-last['step'])*rate;r['last_step_age_seconds']=time.time()-last['time']
 if (p/'gpu.csv').exists():
  lines=list(csv.reader((p/'gpu.csv').read_text().splitlines()));r['gpu_tail']=lines[-24:]
 if (p/'train.log').exists():
  text=(p/'train.log').read_text(errors='replace');r['log_head']=text[:4000];r['log_tail']=text[-500:]
  r['observed_nccl_ranks']=sorted({int(x) for x in re.findall(r'\[(\d+)\] NCCL INFO',text)})
 r['checkpoints']=[{'name':x.name,'complete':(x/'COMPLETE').exists()} for x in p.glob('step-*')]
 r['export_complete']=(p/'TRAIN_COMPLETE').exists()
 if any(not x['complete'] for x in r['checkpoints']):
  r['phase']='saving';r['next_checkpoint_eta_seconds']=None
 else:r['phase']='exported' if r['export_complete'] else ('training' if r.get('last_steps') else 'not_started')
 out['tasks'][task]=r
(B/'status/formal-monitor.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
