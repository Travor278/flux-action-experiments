set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import json,pathlib,importlib.metadata,time
from huggingface_hub import HfApi
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');r={'time':time.time(),'packages':{p:importlib.metadata.version(p) for p in ['huggingface_hub','hf-xet']}}
try:
 a=HfApi().whoami();r.update(authenticated=True,name=a['name'])
except Exception as e:r.update(authenticated=False,error_type=type(e).__name__)
(B/'status/hf-upload-preflight.json').write_text(json.dumps(r,indent=2))
PY
