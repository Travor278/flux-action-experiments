R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/eval150-inventory.json"
from pathlib import Path
import json,time,subprocess
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
print(json.dumps({'time':time.time(),'layouts':{str(seed):{t:len(list((R/f'RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/{seed}').glob(t+'_*.json'))) for t in ['cover_blocks','plug_in_charger']} for seed in range(3)},'git':{n:subprocess.check_output(['git','-C',str(R/n),'rev-parse','HEAD'],text=True).strip() for n in ['RoboDojo','XPolicyLab']}},indent=2))
PY
