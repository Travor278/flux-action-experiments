from pathlib import Path
import json,time,statistics,csv
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
out={'time':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 p=B/'status'/f'pi05_{task}_seed42';r={'complete':(p/'COMPLETE.json').exists(),'ranks':{}}
 for rank in [0,1]:
  f=p/f'progress-rank{rank}.json'
  if f.exists():r['ranks'][rank]=json.loads(f.read_text())
 if (p/'train.log').exists():
  text=(p/'train.log').read_text(errors='replace');r['log_tail']=text[-3000:]
  rows=[]
  for line in text.splitlines():
   if 'PI05_UPDATE ' in line:
    try:
     x=json.loads(line.split('PI05_UPDATE ',1)[1])
     if x['rank']==0:rows.append(x)
    except (ValueError,KeyError):pass
  rows=rows[-20:]
  intervals=[(b['time']-a['time'])/(b['step']-a['step']) for a,b in zip(rows,rows[1:]) if b['step']>a['step']]
  if intervals:
   rate=statistics.median(intervals);last=rows[-1];r.update(seconds_per_step=rate,next_checkpoint_eta_seconds=max(0,min(3000,(last['step']//1000+1)*1000)-last['step'])*rate,last_update_age_seconds=time.time()-last['time'])
 if (p/'gpu.csv').exists():r['gpu_tail']=list(csv.reader((p/'gpu.csv').read_text().splitlines()))[-6:]
 out['tasks'][task]=r
(B/'status/pi05-monitor.json').write_text(json.dumps(out,indent=2))
