R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');r={}
for name in ['apps','source']:
 p=R/name;target=R/'RoboDojo/third_party/IsaacLab'/name
 if p.exists() or p.is_symlink():assert p.resolve()==target.resolve()
 else:p.symlink_to(target,target_is_directory=True)
 r[str(p)]=str(p.resolve())
(R/'status/isaaclab-resource-paths.json').write_text(json.dumps(r,indent=2))
PY