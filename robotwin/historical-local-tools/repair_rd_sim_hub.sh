set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export UV_CACHE_DIR="$R/uv-cache"
printf '\nhuggingface-hub==0.34.6\n' >> "$R/configs/sim-constraints.txt"
uv pip install --python "$R/sim-venv/bin/python" 'huggingface-hub==0.34.6' 'click==8.1.7' 'idna==3.10' > "$R/status/sim-hub-pin-repair.log" 2>&1
uv pip uninstall --python "$R/sim-venv/bin/python" httpx2 httpcore2 >> "$R/status/sim-hub-pin-repair.log" 2>&1
uv pip check --python "$R/sim-venv/bin/python" >> "$R/status/sim-hub-pin-repair.log" 2>&1
"$R/sim-venv/bin/python" - <<'PY' >> "$R/status/sim-hub-pin-repair.log" 2>&1
from pathlib import Path
import time,json,importlib.metadata as m
from curobo.batch_motion_planner import BatchMotionPlanner,MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
r={'time':time.time(),'packages':{n:m.version(n) for n in ['torch','isaacsim','isaaclab','numpy','warp-lang','click','idna','huggingface-hub']},'curobo_import':True,'renderer_smoke':False}
assert r['packages']['torch'].startswith('2.7.0');(R/'status/sim-extensions-installed.json').write_text(json.dumps(r,indent=2))
PY