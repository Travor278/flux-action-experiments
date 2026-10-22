set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
export HF_XET_CACHE="$S/hf-transfer-cache"
unset HF_XET_HIGH_PERFORMANCE HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_FIXED_UPLOAD_CONCURRENCY=8
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/upload_flux_hf.py" > "$B/status/hf-final-upload.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/hf-final-upload.pid"
