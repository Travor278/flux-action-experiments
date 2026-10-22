set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec > "$R/status/ws-deps-install.log" 2>&1
export UV_CACHE_DIR="$R/uv-cache"
"$R/sim-venv/bin/python" - <<'PY'
from pathlib import Path
import importlib.metadata as m,subprocess,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
names=['pydantic','pydantic-core','annotated-types','typing-extensions','typing-inspection']
pins=[name+'=='+m.version(name) for name in names]
print(pins,flush=True)
subprocess.run(['uv','pip','install','--python',str(R/'policy-venv/bin/python'),'--target',str(R/'ws-deps'),'--no-deps',*pins],check=True,timeout=240)
(R/'status/ws-deps-install.json').write_text(json.dumps({'time':time.time(),'pins':pins,'target':str(R/'ws-deps'),'active_environment_modified':False},indent=2))
PY
