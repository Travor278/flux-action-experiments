set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/flux-action"
OMP_NUM_THREADS=2 "$R/policy-venv/bin/python" -m pytest tests/test_history_training.py tests/test_history_inference.py "$R/scripts/test_temporal_policy_integration.py" -q > "$R/status/temporal-policy-integration-tests.log" 2>&1
printf PASS > "$R/status/temporal-policy-integration-tests-passed.txt"