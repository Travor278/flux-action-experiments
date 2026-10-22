set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
from pathlib import Path
import json,os,subprocess,time
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=Path(str(B).replace('/hdd/','/ssd/'))
r={'time':time.time(),'roots':{},'processes':[]}
for root in [B,S,B/'assets',S/'runs']:
 r['roots'][str(root)]=[{'name':p.name,'symlink':p.is_symlink(),'target':str(p.resolve()),'du':subprocess.run(['du','-s','-B1',str(p)],capture_output=True,text=True).stdout.strip()} for p in root.iterdir()]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(B) in cmd or str(S) in cmd:r['processes'].append({'pid':p.name,'cmd':cmd[:1200]})
 except OSError:pass
for n in ['hf-tail-uploads.json','hf-tail-upload.pid']:
 p=B/'status'/n;r[n]=p.read_text() if p.exists() else None
r['upload_log_tail']=(B/'status/hf-tail-upload.log').read_text(errors='replace')[-1500:]
(B/'status/robotwin-close-inventory.json').write_text(json.dumps(r,indent=2))
PY
