set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/'runs/two-gpu-efficiency-sweep-v1'
p=B/'status/efficiency-sweep.json';out={'time':time.time(),'report':json.loads(p.read_text()) if p.exists() else None}
current=(out['report'] or {}).get('current')
if current:
 d=R/('selected-validation' if current=='selected_save_resume_test' else current)
 for name in ['train','resume','safe-stop','export']:
  p=d/f'{name}.log'
  if p.exists():
   s=p.read_text(errors='replace');out[name+'_tail']=s[-1800:]
   if 'Traceback' in s:out[name+'_error']=s[s.index('Traceback'):][:3500]
 p=d/'metrics.jsonl'
 if p.exists():
  rows=[]
  for line in p.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
  out['metrics_tail']=rows[-3:]
out['verified']=(B/'status/efficiency-selected-verified.json').exists()
(B/'status/efficiency-monitor.json').write_text(json.dumps(out,indent=2))
PY
