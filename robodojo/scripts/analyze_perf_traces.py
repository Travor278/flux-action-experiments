"""Read-only union-of-kernel-time analysis; annotations are not GPU work."""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,json,time,statistics
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
run=Path(str(R).replace('/hdd/','/ssd/'))/'runs/perf-exact-20261014'
results=[]
for mode in ['baseline','cached']:
 for rank in (0,1):
  p=run/mode/f'rank{rank}-trace.json.gz'
  with gzip.open(p,'rt') as f:j=json.load(f)
  events=[e for e in j['traceEvents'] if e.get('ph')=='X' and e.get('dur',0)>0]
  categories=Counter(e.get('cat','') for e in events)
  gpu=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
  assert gpu,categories
  spans=sorted((e['ts'],e['ts']+e['dur']) for e in gpu);merged=[]
  for a,b in spans:
   if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
   else:merged.append([a,b])
  envelope=(merged[-1][1]-merged[0][0])/1e6;busy=sum(b-a for a,b in merged)/1e6
  gaps=sorted([(b[0]-a[1],a[1],b[0]) for a,b in zip(merged,merged[1:])],reverse=True)[:8]
  annotated=[e for e in events if e.get('cat')=='user_annotation']
  gap_rows=[]
  for length,a,b in gaps:
   midpoint=(a+b)/2
   active=sorted({e['name'] for e in annotated if e['ts']<=midpoint<=e['ts']+e['dur']})
   gap_rows.append({'seconds':length/1e6,'cpu_annotations_at_midpoint':active})
  summed=defaultdict(float)
  for e in gpu:
   n=e['name'];key='communication' if 'nccl' in n.lower() else 'copy' if e.get('cat')=='gpu_memcpy' else 'other_kernel'
   summed[key]+=e['dur']/1e6
  report=json.loads((run/mode/f'comparison-rank{rank}.json').read_text())
  normal=[e for e in report['updates'][2:] if not e['profiled']]
  results.append({'mode':mode,'rank':rank,'trace_bytes':p.stat().st_size,'categories':dict(categories),'gpu_envelope_seconds':envelope,'gpu_busy_union_seconds':busy,'gpu_idle_union_seconds':envelope-busy,'busy_fraction':busy/envelope,'summed_activity_seconds_overlaps_allowed':dict(summed),'largest_idle_gaps':gap_rows,'normal_update_median':statistics.median(e['metrics']['step_time'] for e in normal),'host_phase_medians':{k:statistics.median(e['cpu_host_seconds'][k] for e in normal) for k in normal[0]['cpu_host_seconds']}})
 del j,events,gpu,spans,annotated
(R/'status/perf-trace-analysis.json').write_text(json.dumps({'time':time.time(),'results':results,'scope':'One profiled update per mode/rank; kernel/copy interval unions exclude GPU annotation ranges. Small sample, diagnostic overhead; not production-wide GPU utilization.'},indent=2))
