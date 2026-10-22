set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
if kill -0 "$(cat "$B/status/hf-tail-upload.pid")" 2>/dev/null; then exit 23; fi
cp "$B/status/hf-tail-upload.log" "$B/status/hf-tail-upload-attempt1.log"
unset HF_XET_HIGH_PERFORMANCE HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_FIXED_UPLOAD_CONCURRENCY=8 HF_XET_CACHE=/workspace/ssd/codex_remote_ops/flux_piperx_20261008/hf-tail-transfer-cache
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/upload_tail_hf.py" > "$B/status/hf-tail-upload.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/hf-tail-upload.pid"