set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/inspect_robodojo_v3.py" > "$R/status/v3-inspection.log" 2>&1 < /dev/null &
mkdir -p "$R/XPolicyLab/policy/FLUX_RoboDojo_History"
cd "$R/flux-action"
nohup env PYTHONPATH="$R/flux-action/src" CUDA_VISIBLE_DEVICES='' "$B/flux-action/.venv/bin/python" -m pytest tests/test_history_inference.py tests/test_history_training.py tests/test_so101_config.py -q > "$R/status/official-history-tests.log" 2>&1 < /dev/null &