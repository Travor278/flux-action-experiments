set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,hashlib
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1';p=out/'train.log';lines=p.read_text(errors='replace').splitlines();hits=[]
for i,line in enumerate(lines):
 if 'Traceback' in line or 'Error:' in line or 'FULL_PREFIX' in line or 'FULL_PREFLIGHT' in line:
  hits.append({'line':i+1,'text':'\n'.join(lines[max(0,i-3):i+24])})
r={'time':time.time(),'log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'matches':hits,'state2':json.loads((out/'step-2/state.json').read_text()),'fingerprint_files':[str(p.name) for p in out.glob('exact-*')]}
(R/'status/full-preflight-first-failure.json').write_text(json.dumps(r,indent=2))
PY