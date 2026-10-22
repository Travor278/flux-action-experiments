set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B/RoboTwin"
export PYTHONPATH="$B/RoboTwin:$B/XPolicyLab:$B:${PYTHONPATH:-}"
"$B/sim-venv/bin/python" - <<'PY' > "$B/status/sim-import-check.tmp"
import torch,numpy,sapien,mplib
import envs.click_bell,envs.handover_block
print('SIM_IMPORTS_OK',torch.__version__,numpy.__version__)
PY
mv "$B/status/sim-import-check.tmp" "$B/status/sim-import-check.txt"
