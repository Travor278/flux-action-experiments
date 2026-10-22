set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" "$B/scripts/cleanup_hf_upload_staging.py" > "$B/status/hf-upload-staging-cleanup.log" 2>&1
