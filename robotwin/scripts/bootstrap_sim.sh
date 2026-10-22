#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
export UV_CACHE_DIR=/tmp/flux-piperx-sim-uv
export UV_PYTHON_INSTALL_DIR="$B/python"
cd "$B"
uv venv --python 3.10 "$B/sim-venv"
uv pip install --python "$B/sim-venv/bin/python" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$B/sim-venv/bin/python" -r "$B/scripts/sim-requirements.txt"
uv pip freeze --python "$B/sim-venv/bin/python" > "$B/status/sim-pip-freeze.txt"
export PYTHONPATH="$B/RoboTwin:$B/XPolicyLab:$B:${PYTHONPATH:-}"
"$B/sim-venv/bin/python" - <<'PY' > status/sim-import-check.tmp
import torch,numpy,sapien,mplib
import envs.click_bell,envs.handover_block
print('SIM_IMPORTS_OK',torch.__version__,numpy.__version__)
PY
mv status/sim-import-check.tmp status/sim-import-check.txt
