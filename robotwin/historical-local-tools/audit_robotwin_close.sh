set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
from pathlib import Path
import os,json,time
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=Path(str(B).replace('/hdd/','/ssd/'))
r={'time':time.time(),'remaining_experiment_weights':[],'deleted_paths_absent':True}
proof=json.loads((B/'status/robotwin-close-cleanup.json').read_text());assert proof['complete']
for x in proof['deleted']:assert not Path(x['path']).exists()
for root in [B/'runs',B/'RoboTwin',B/'evaluation']:
 for d,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[n for n in dirs if n not in {'.git','__pycache__','.cache'} and not (Path(d)/n).is_symlink()]
  for n in files:
   p=Path(d)/n
   if p.is_symlink():continue
   if p.suffix in {'.safetensors','.distcp','.ckpt','.pth','.pt'}:r['remaining_experiment_weights'].append(str(p))
assert not r['remaining_experiment_weights']
r['verified']=True;(B/'status/robotwin-close-residual-audit.json').write_text(json.dumps(r,indent=2))
PY