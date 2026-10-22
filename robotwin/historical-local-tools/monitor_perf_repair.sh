set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time,statistics
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/'runs/two-gpu-perf-repair-v1'
out={'time':time.time(),'exists':R.exists(),'phases':{}}
for rank in [0,1]:
 p=R/f'phases-rank{rank}.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
 out['phases'][str(rank)]=rows[-3:]
 for mode in ['baseline','buffered']:
  part=[x for x in rows if x['mode']==mode and x['step']>1]
  if part:out[f'rank{rank}_{mode}']={'updates':len(part),'seconds':statistics.median(x['update_seconds'] for x in part),'phases':{k:statistics.median(x['phases'][k] for x in part) for k in part[0]['phases']}}
for name in ['train','resume','export']:
 p=R/f'{name}.log'
 if p.exists():
  text=p.read_text(errors='replace');out[name+'_tail']=text[-1800:]
  if 'Traceback' in text:out[name+'_error']=text[text.index('Traceback'):][:4500]
out['checkpoints']=[{'name':p.name,'complete':(p/'COMPLETE').exists()} for p in R.glob('step-*')]
if (R/'metrics.jsonl').exists():out['metrics_tail']=[json.loads(x) for x in (R/'metrics.jsonl').read_text().splitlines()][-3:]
p=B/'status/two-gpu-perf-repair-result.json';out['result']=json.loads(p.read_text()) if p.exists() else None
(B/'status/perf-repair-monitor.json').write_text(json.dumps(out,indent=2))
PY
