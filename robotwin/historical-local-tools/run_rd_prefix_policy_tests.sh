set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/scripts"
"$R/policy-venv/bin/python" install_prefix_policy.py > "$R/status/prefix-policy-install.log" 2>&1
OMP_NUM_THREADS=2 "$R/policy-venv/bin/python" -m pytest "$R/scripts/test_prefix_policy.py" "$R/scripts/test_temporal_policy_integration.py" "$R/flux-action/tests/test_history_training.py" "$R/flux-action/tests/test_history_inference.py" -q > "$R/status/prefix-policy-tests.log" 2>&1
printf PASS > "$R/status/prefix-policy-tests-passed.txt"