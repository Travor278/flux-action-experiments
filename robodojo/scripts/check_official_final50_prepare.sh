set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec > "$R/status/official-final50-prepare.log" 2>&1
bash -n "$R/scripts/launch_official_final50.sh"
bash -n "$R/scripts/launch_official_final50_pair.sh"
"$R/policy-venv/bin/python" "$R/scripts/prepare_official_final50_plan.py"
PYTHONPATH="$R/RoboDojo:$R/XPolicyLab:$R" "$R/sim-venv/bin/python" - <<'PY'
from XPolicyLab.policy.FLUX_RoboDojo_Official.deploy import eval_one_episode
print('FINAL_DEPLOY_IMPORT_PASS_NO_GPU')
PY

"$R/policy-venv/bin/python" "$R/scripts/monitor_official_history.py"
