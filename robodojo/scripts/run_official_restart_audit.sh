set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/scripts:$R/flux-action-official-v1/src" PYTHONNOUSERSITE=1
nohup "$R/policy-venv/bin/python" "$R/scripts/audit_official_history_restart.py" > "$R/status/official-history-restart-audit-20261015.log" 2>&1 < /dev/null &
