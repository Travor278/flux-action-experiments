set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" "$B/scripts/check_hf_remote_only.py" > "$B/status/hf-checkpoint-only-verify.log" 2>&1
