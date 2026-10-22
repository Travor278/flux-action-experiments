set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec 9> "$R/status/sim-install.lock"
flock -n 9 || exit 0
export UV_CACHE_DIR="$R/uv-cache" UV_PYTHON_INSTALL_DIR="$R/python" PYTHONNOUSERSITE=1
if [ ! -x "$R/sim-venv/bin/python" ]; then uv python install 3.11; fi
if [ ! -x "$R/sim-venv/bin/python" ]; then uv venv --python 3.11 "$R/sim-venv"; fi
uv pip install --python "$R/sim-venv/bin/python" 'torch==2.7.0' 'torchvision==0.22.0' 'torchaudio==2.7.0' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$R/sim-venv/bin/python" 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
uv pip install --python "$R/sim-venv/bin/python" 'numpy==1.26.0' 'packaging==23.0' 'typing_extensions==4.12.2' 'filelock==3.13.1' 'websockets==12.0' 'scipy==1.15.3' 'warp-lang==1.11.0' 'click==8.1.7' 'psutil==5.9.8' 'wheel==0.45.1' 'starlette==0.45.3' 'onnx>=1.18,<1.22' 'ipython<9' 'virtualenv==20.30.0'
uv pip install --python "$R/sim-venv/bin/python" -r "$R/RoboDojo/scripts/requirements.txt" 'opencv-python-headless==4.11.0.86' pillow matplotlib scikit-learn
"$R/sim-venv/bin/python" - <<'PY'
import sys,json,time,importlib.metadata as m,pathlib
R=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
names=['torch','isaacsim','numpy','websockets'];r={'time':time.time(),'python':sys.version,'packages':{n:m.version(n) for n in names},'stage':'base_packages_installed','simulator_smoke_passed':False,'remaining':['IsaacLab','CuRobo','assets','headless renderer smoke']}
(R/'status/sim-base-installed.json').write_text(json.dumps(r,indent=2))
PY
