R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/scene-attempt2-evidence.log" 2>&1
from pathlib import Path
import json,hashlib,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
c=json.loads((R/'status/scene-control-current.json').read_text());assert 'job-fcee60a5-' in c['hostname'];run=Path(c['run'])
for n in ['probe_scene_control.py','launch_scene_control.sh']:(run/('original-'+n)).write_bytes((R/'scripts'/n).read_bytes())
failure=json.loads((run/'cover_blocks/failure.json').read_text());assert failure['type']=='ModuleNotFoundError' and "'client_server'" in failure['message']
r={'time':time.time(),'job':c['hostname'],'rootcause':failure,'log_sha256':hashlib.sha256((run/'cover_blocks.log').read_bytes()).hexdigest(),'repair':'Add R/XPolicyLab to task PYTHONPATH, matching official eval client launcher; no model/data/task physics change'}
(R/'status/scene-module-path-repair.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
PY
cd "$R/RoboDojo"
PYTHONPATH="$R/RoboDojo:$R/XPolicyLab:$R:$R/scripts" "$R/sim-venv/bin/python" - <<'PY' > "$R/status/scene-client-import-test.log" 2>&1
from client_server.ws.model_client import WsModelClient
from XPolicyLab.utils.process_data import pack_robot_state,unpack_robot_state
from XPolicyLab.policy.FLUX_RoboDojo_History.model import image_chw
import numpy as np
info={'arm_dim':[6,6],'ee_dim':[1,1]};state=np.arange(14,dtype=np.float32)
a=unpack_robot_state(state,'joint',info);np.testing.assert_array_equal(pack_robot_state({'state':a},'joint',info),state)
print('SIM_CLIENT_MODULE_AND_STATE_ROUNDTRIP_PASS',flush=True)
PY
