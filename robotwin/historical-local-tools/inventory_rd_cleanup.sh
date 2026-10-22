set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import os,json,time,subprocess
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
r={'time':time.time(),'candidates':[],'runs':{},'processes':[]}
for root in [R,S]:
 if not root.exists():continue
 for p in root.iterdir():
  if p.is_file() and (p.name=='core' or p.name.startswith('core.') or p.name.endswith(('.tar','.tar.gz','.partial','.tmp'))):
   r['candidates'].append({'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime})
 for p in (root/'runs').glob('*'):
  if p.is_dir():
   r['runs'][str(p)]={'checkpoints':[str(c.name) for c in p.glob('step-*')],'complete':[str(c.relative_to(p)) for c in p.glob('step-*/COMPLETE')]}
p=R/'runs/checkpoint-cpu-smoke'
if p.exists():r['candidates'].append({'path':str(p),'bytes':sum(f.stat().st_size for f in p.rglob('*') if f.is_file()),'reason':'Finished standaloneCPU checkpoint regression, GPU diagnostics use SSD separate path'})
ps=subprocess.check_output(['ps','-ww','-eo','pid,args'],text=True)
r['processes']=[l for l in ps.splitlines() if str(R) in l]
(R/'status/cleanup-inventory-current.json').write_text(json.dumps(r,indent=2))
PY