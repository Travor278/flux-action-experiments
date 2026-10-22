set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/sim-venv/bin/python" - <<'PY'
from pathlib import Path
import time,json,importlib.metadata as m
from curobo.batch_motion_planner import BatchMotionPlanner,MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
r={'time':time.time(),'packages':{n:m.version(n) for n in ['torch','isaacsim','isaaclab','numpy','warp-lang','click','idna','huggingface-hub']},'curobo_import':True,'renderer_smoke':False,'metadata_compatibility_record':'isaaclab-install-metadata-compat.json'}
assert r['packages']['torch'].startswith('2.7.0');(R/'status/sim-extensions-installed.json').write_text(json.dumps(r,indent=2))
PY
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1
nohup bash -c 'R="$1"; timeout --signal=TERM --kill-after=15s 300s "$R/sim-venv/bin/python" "$R/scripts/render_smoke.py" > "$R/status/renderer-smoke.log" 2>&1; echo "$?" > "$R/status/renderer-smoke.exitcode"' bash "$R" > "$R/status/renderer-launch.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/renderer-smoke.pid"