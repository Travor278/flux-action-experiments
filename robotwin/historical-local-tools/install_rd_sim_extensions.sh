set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec 9> "$R/status/sim-extension-install.lock"
flock -n 9 || exit 0
test -f "$R/status/sim-base-installed.json"
test "$(git -C "$R/RoboDojo/third_party/IsaacLab" rev-parse HEAD)" = afca7b09d60d8beb9c1cb28b43066499940b969b
test "$(git -C "$R/RoboDojo/third_party/curobo" rev-parse HEAD)" = d17b54ce32cba095c0b000c4c58777075d11de0e
unset CONDA_PREFIX CONDA_DEFAULT_ENV
source "$R/sim-venv/bin/activate"
export UV_CACHE_DIR="$R/uv-cache" PYTHONNOUSERSITE=1 TERM=xterm-256color MAX_JOBS=4 OMNI_KIT_ACCEPT_EULA=YES
cat > "$R/configs/sim-constraints.txt" <<'EOF'
torch==2.7.0
torchvision==0.22.0
torchaudio==2.7.0
numpy==1.26.0
packaging==23.0
typing_extensions==4.12.2
filelock==3.13.1
websockets==12.0
scipy==1.15.3
warp-lang==1.11.0
click==8.1.7
idna==3.10
huggingface-hub==0.34.6
psutil==5.9.8
wheel==0.45.1
starlette==0.45.3
onnx>=1.18,<1.22
ipython<9
virtualenv==20.30.0
EOF
export UV_CONSTRAINT="$R/configs/sim-constraints.txt"
uv pip install --python "$R/sim-venv/bin/python" pip
cd "$R/RoboDojo/third_party/IsaacLab"
bash isaaclab.sh --install none
cd "$R/RoboDojo/third_party/curobo"
uv pip install --python "$R/sim-venv/bin/python" -e '.[cu12]' --no-build-isolation -c "$R/configs/sim-constraints.txt"
uv pip check --python "$R/sim-venv/bin/python"
python - <<'PY'
from pathlib import Path
import time,json,importlib.metadata as m
from curobo.batch_motion_planner import BatchMotionPlanner,MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
r={'time':time.time(),'packages':{n:m.version(n) for n in ['torch','isaacsim','isaaclab','numpy','warp-lang']},'curobo_import':True,'renderer_smoke':False}
assert r['packages']['torch'].startswith('2.7.0');(R/'status/sim-extensions-installed.json').write_text(json.dumps(r,indent=2))
PY
