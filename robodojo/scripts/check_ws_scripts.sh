#!/usr/bin/env bash
set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec > "$R/status/ws-control-cpu-check.log" 2>&1
export PYTHONPATH="$R/ws-deps:$R/scripts:$R/flux-action/src:$R/XPolicyLab:$R" PYTHONNOUSERSITE=1
bash -n "$R/scripts/launch_ws_control.sh"
"$R/policy-venv/bin/python" - <<'PY'
import ast,json,hashlib,time
from pathlib import Path
from client_server.ws.model_server import PolicyServer,PolicyServerConfig
from XPolicyLab.policy.FLUX_RoboDojo_History.model import Model
from flux_action.config import PolicyConfig
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
gate=json.loads((R/'status/ws-control-ready.json').read_text())
for name,expected in gate['files'].items():
 p=R/'scripts'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
 if name.endswith('.py'):ast.parse(p.read_text())
export=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1/export-4-ema'
c=json.loads((export/'config.json').read_text())
assert c['n_action_steps']==16 and c['n_obs_steps']==8 and c['action_dim']==14 and c['fps']==25
assert c['prefix_memory_spec'] and c['torch_dtype']=='bfloat16',c['torch_dtype']
(R/'status/ws-control-cpu-verified.json').write_text(json.dumps({'time':time.time(),'verified':False,'server_import':True,'dtype':c['torch_dtype'],'files':gate['files'],'scope':'Imports, syntax, export configuration and artifact hashes only; not real WS or GPU simulation proof'},indent=2))
print('WS_CPU_IMPORT_CONFIG_PASS',flush=True)
PY
PYTHONPATH="$R/scripts:$R/XPolicyLab:$R" "$R/sim-venv/bin/python" - <<'PY'
from client_server.ws.model_client import WsModelClient
from XPolicyLab.policy.FLUX_RoboDojo_History.model import Model
from pathlib import Path
import json,time
p=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013/status/ws-control-cpu-verified.json')
r=json.loads(p.read_text());r.update(verified=True,sim_client_import=True,finished=time.time());p.write_text(json.dumps(r,indent=2))
print('SIM_WS_CLIENT_ADAPTER_IMPORT_PASS',flush=True)
PY
"$R/policy-venv/bin/python" "$R/scripts/monitor_formal_training.py"
