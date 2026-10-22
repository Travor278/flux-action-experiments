set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export UV_CACHE_DIR="$R/uv-cache"
uv pip install --python "$R/policy-venv/bin/python" h5py
cd "$R/flux-action"
PYTHONPATH="$R:$R/flux-action/src" FLUX_ADAPTER_PATH="$R/XPolicyLab/policy/FLUX_RoboDojo_History/model.py" CUDA_VISIBLE_DEVICES='' "$R/policy-venv/bin/python" -m pytest tests/test_history_inference.py tests/test_history_training.py tests/test_so101_config.py "$R/scripts/test_adapter.py" -q > "$R/status/new-environment-tests.log" 2>&1
uv pip check --python "$R/policy-venv/bin/python" > "$R/status/policy-pip-check.log" 2>&1