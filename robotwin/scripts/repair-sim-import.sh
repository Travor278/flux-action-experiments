set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
export UV_CACHE_DIR=/tmp/flux-piperx-sim-uv
uv pip install --python "$B/sim-venv/bin/python" setuptools==69.5.1
export PYTHONPATH="$B/RoboTwin:$B/XPolicyLab:$B:${PYTHONPATH:-}"
"$B/sim-venv/bin/python" - <<'PY' > status/sim-import-check.tmp
import torch,numpy,sapien,mplib
import envs.click_bell,envs.handover_block
print('SIM_IMPORTS_OK',torch.__version__,numpy.__version__)
PY
mv status/sim-import-check.tmp status/sim-import-check.txt
