set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
PYTHONPATH="$R:$R/flux-action/src" FLUX_ADAPTER_PATH="$R/XPolicyLab/policy/FLUX_RoboDojo_History/model.py" CUDA_VISIBLE_DEVICES='' "$B/flux-action/.venv/bin/python" -m pytest "$R/scripts/test_adapter.py" -q > "$R/status/adapter-tests.log" 2>&1