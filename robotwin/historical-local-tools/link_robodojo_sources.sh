set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$B/flux-action/.venv/bin/python" - <<'PY'
from pathlib import Path
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
for p,target in [(R/'env_cfg',R/'RoboDojo/env_cfg'),(R/'RoboDojo/XPolicyLab',R/'XPolicyLab')]:
 if p.is_symlink():assert p.resolve()==target.resolve();continue
 if p.exists():p.rmdir()
 p.symlink_to(target,target_is_directory=True)
PY
PYTHONPATH="$R" "$B/flux-action/.venv/bin/python" - <<'PY' > "$R/status/robot-action-contract.json"
import json
from XPolicyLab.utils.process_data import get_robot_action_dim_info
x=get_robot_action_dim_info('arx_x5');assert x=={'arm_dim':[6,6],'ee_dim':[1,1]};print(json.dumps({'verified':True,'dimensions':x,'order':['left_arm_6','left_gripper_1','right_arm_6','right_gripper_1']}))
PY