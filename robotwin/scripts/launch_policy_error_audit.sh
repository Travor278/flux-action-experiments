#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
exec 9>"$B/status/policy-error-audit.lock"
flock -n 9 || exit 3
cd "$B"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$B/flux-action/src:$B:${PYTHONPATH:-}"
test -s "$B/status/policy-error-audit-ready.json"
nvidia-smi > "$B/status/policy-error-audit-gpu-preflight.txt"
"$B/flux-action/.venv/bin/python" "$B/scripts/run_policy_error_audit.py" > "$B/status/policy-error-audit.log" 2>&1
