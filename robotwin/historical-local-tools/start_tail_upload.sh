set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
unset HF_XET_HIGH_PERFORMANCE HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_FIXED_UPLOAD_CONCURRENCY=8 HF_XET_CACHE="$S/hf-tail-transfer-cache"
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/upload_tail_hf.py" > "$B/status/hf-tail-upload.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/hf-tail-upload.pid"
