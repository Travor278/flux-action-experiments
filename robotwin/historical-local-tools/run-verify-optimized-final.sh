set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" "$B/scripts/verify_optimized_final.py" > "$B/status/optimized-final-verify.log" 2>&1
