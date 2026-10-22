R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');B=R.parent/'flux_piperx_20261008'
found=[]
for base in (R,B):
 for p in base.glob('*/lib/python3.12/site-packages/pydantic'):
  site=p.parent
  found.append({'site':str(site),'metadata':[x.name for x in site.glob('pydantic*.dist-info')],'binary':[x.name for x in (site/'pydantic_core').glob('*.so')]})
(R/'status/ws-offline-dependency-candidates.json').write_text(json.dumps({'time':time.time(),'candidates':found},indent=2))
PY
