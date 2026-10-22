R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');root=Path('/usr/lib/x86_64-linux-gnu')
r={n:str((root/n).resolve()) for n in ['libnvoptix.so.1','libGLX_nvidia.so.0','libEGL_nvidia.so.0','libcuda.so.1','libnvidia-ml.so.1']}
(R/'status/nvidia-link-audit.json').write_text(json.dumps(r,indent=2))
PY