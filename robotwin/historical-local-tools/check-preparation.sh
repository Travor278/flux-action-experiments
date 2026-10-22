set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,json,time,os,re
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
out={'time':time.time(),'downloads':[],'conversions':{},'bootstrap':{}}
for folder in [B/'assets/.flux-base.download',B/'assets/flux-base']:
 for p in folder.rglob('*'):
  if p.is_file() and ('incomplete' in p.name or p.suffix=='.safetensors'):out['downloads'].append({'path':str(p.relative_to(B)),'bytes':p.stat().st_size,'modified':p.stat().st_mtime})
for t in ['click_bell','handover_block']:
 p=B/'assets/lerobot'/t/'CONVERSION_COMPLETE.json'
 if p.exists():
  a=json.loads(p.read_text());lens=[x['frames'] for x in a['episodes']]
  out['conversions'][t]={'episodes':len(lens),'frames':sum(lens),'valid_windows':sum(max(0,n-33) for n in lens)}
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  if b'uv sync --project flux-action' not in p.read_bytes().replace(b'\0',b' '):continue
  out['bootstrap'][p.parent.name]={'io':(p.parent/'io').read_text(),'files':[]}
  for f in (p.parent/'fd').iterdir():
   try:
    link=os.readlink(f)
    if '/tmp/flux-piperx' in link:out['bootstrap'][p.parent.name]['files'].append({'path':link,'size':f.stat().st_size})
   except OSError:pass
 except OSError:pass
logs=sorted(pathlib.Path('/tmp/flux-piperx-xet').rglob('*.log'))
if logs:
 lines=logs[-1].read_text(errors='replace').splitlines()
 out['xet_progress']=[re.sub(r'https?://[^\s"<>]+','<URL>',x) for x in lines if 'Current concurrency' in x][-2:]
(B/'status/preparation-summary.json').write_text(json.dumps(out,indent=2))
PY
