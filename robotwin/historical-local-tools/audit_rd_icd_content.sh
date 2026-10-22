R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
out={str(p):p.read_text() for p in [Path('/etc/vulkan/icd.d/nvidia_icd.json'),Path('/usr/share/vulkan/icd.d/nvidia_icd.json')]}
s=(R/'status/renderer-smoke-attempt3.log').read_text(errors='replace');out['errors']=[x for x in s.splitlines() if '[Error]' in x or 'VkResult' in x or 'Vulkan' in x][:30]
(R/'status/vulkan-icd-audit.json').write_text(json.dumps(out,indent=2))
PY