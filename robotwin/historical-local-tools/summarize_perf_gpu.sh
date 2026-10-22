set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,csv,datetime,statistics,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/'runs/two-gpu-perf-repair-v1'
m={r['step']:r for r in map(json.loads,(R/'metrics.jsonl').read_text().splitlines()) if 'step_time' in r}
out={'time':time.time(),'modes':{}}
for mode,lo,hi in [('baseline',2,5),('buffered',8,20)]:
 telemetry={0:[],1:[]};start=m[lo-1]['time'];end=m[hi]['time']
 for row in csv.reader((R/'gpu.csv').read_text().splitlines()):
  try:
   t=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
   if start<=t<=end:telemetry[int(row[1])].append(float(row[2]))
  except (ValueError,IndexError):pass
 rate=statistics.median(m[i]['time']-m[i-1]['time'] for i in range(lo,hi+1))
 out['modes'][mode]={'wall_seconds_per_update':rate,'windows_per_second':32/rate,'gpu_util_mean':{str(k):statistics.mean(v) for k,v in telemetry.items() if v},'samples':{str(k):len(v) for k,v in telemetry.items()}}
(B/'status/perf-gpu-comparison.json').write_text(json.dumps(out,indent=2))
PY
