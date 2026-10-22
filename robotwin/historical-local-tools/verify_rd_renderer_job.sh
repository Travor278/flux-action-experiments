set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
bash -n "$R/scripts/launch_renderer_job.sh"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import ast,json,hashlib,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');p=R/'scripts/render_smoke_job_v3.py';ast.parse(p.read_text())
assert (R/'status/sim-extensions-installed.json').exists()
r={'time':time.time(),'ready':True,'scope':'one4090driver/RGBpreflight only','files':{n:hashlib.sha256((R/'scripts'/n).read_bytes()).hexdigest() for n in ['launch_renderer_job.sh','render_smoke_job_v3.py']}}
(R/'status/renderer-job-ready.json').write_text(json.dumps(r,indent=2))
PY