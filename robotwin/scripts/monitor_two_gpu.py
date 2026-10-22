from pathlib import Path
import json,time,csv,statistics
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
R=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008/runs/two-gpu-feasibility-v1')
out={'time':time.time(),'run_exists':R.exists()}
for name,p in [('gpu_unit',B/'status/cpu-ema-gpu-test.log'),('train',R/'train.log'),('resume',R/'resume.log'),('export',R/'export.log')]:
 if p.exists():
  text=p.read_text(errors='replace');out[name+'_tail']=text[-3500:]
  if name in ('train','resume'):
   ready=[]
   for line in text.splitlines():
    if line.startswith('{') and 'TWO_GPU_RANK_READY' in line:
     try: ready.append(json.loads(line))
     except json.JSONDecodeError: pass
   out[name+'_ranks']=ready
  if 'Traceback' in text:
   i=text.find('Traceback');out[name+'_first_error']=text[max(0,i-200):i+5500]
if (R/'metrics.jsonl').exists():
 rows=[]
 for line in (R/'metrics.jsonl').read_text().splitlines():
  try: rows.append(json.loads(line))
  except json.JSONDecodeError: pass
 out['last_metrics']=rows[-3:]
 steps=[x for x in rows if 'step_time' in x and x.get('step',0)>=5]
 intervals=[b['time']-a['time'] for a,b in zip(steps,steps[1:]) if b['step']==a['step']+1 and b['step']<=12]
 if intervals: out['steady_seconds_per_update']=statistics.median(intervals)
if (R/'gpu.csv').exists():
 telemetry=[]
 for row in list(csv.reader((R/'gpu.csv').read_text().splitlines()))[-120:]:
  try: telemetry.append({'time':row[0].strip(),'gpu':int(row[1]),'util':float(row[2]),'memory_mib':float(row[3]),'power_w':float(row[4])})
  except (ValueError,IndexError): pass
 out['recent_gpu']={str(i):{'samples':len(xs),'mean_util':statistics.mean(x['util'] for x in xs),'max_memory_mib':max(x['memory_mib'] for x in xs),'last':xs[-1]} for i in sorted({x['gpu'] for x in telemetry}) if (xs:=[x for x in telemetry if x['gpu']==i])}
out['checkpoints']=[{'name':p.name,'complete':(p/'COMPLETE').exists()} for p in R.glob('step-*')]
proof=B/'status/two-gpu-feasibility-verified.json'
out['verified']=json.loads(proof.read_text()) if proof.exists() else None
(B/'status/two-gpu-monitor.json').write_text(json.dumps(out,indent=2))
