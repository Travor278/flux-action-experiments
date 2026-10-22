set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
bash -n "$R/scripts/launch_export_inference.sh"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,ast,hashlib
from flux_action.policy import FluxActionPolicy
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
assert json.loads((R/'status/full-preflight-export-audit.json').read_text())['verified']
assert hasattr(FluxActionPolicy,'set_text_encoder_offload')
ast.parse((R/'scripts/probe_export_inference.py').read_text())
(R/'status/export-inference-ready.json').write_text(json.dumps({'time':time.time(),'ready':True,'files':{n:hashlib.sha256((R/'scripts'/n).read_bytes()).hexdigest() for n in ['launch_export_inference.sh','probe_export_inference.py']},'scope':'Readonly real export CUDA inference, no formaltraining/no successrate'},indent=2))
PY