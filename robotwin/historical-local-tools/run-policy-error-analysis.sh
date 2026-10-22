set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" "$B/scripts/analyze_policy_error_audit.py" > "$B/status/policy-error-analysis.log" 2>&1
