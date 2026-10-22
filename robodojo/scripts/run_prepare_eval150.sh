set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/scripts:$R/RoboDojo:$R/XPolicyLab:$R" PYTHONNOUSERSITE=1
unset EVAL_NUM
bash -n "$R/scripts/launch_r2_eval150.sh"
bash -n "$R/scripts/launch_r2_eval150_pair.sh"
"$R/policy-venv/bin/python" "$R/scripts/test_eval150_contract.py" > "$R/status/eval150-tests.log" 2>&1
"$R/sim-venv/bin/python" "$R/scripts/prepare_r2_eval150.py" > "$R/status/eval150-prepare.log" 2>&1
