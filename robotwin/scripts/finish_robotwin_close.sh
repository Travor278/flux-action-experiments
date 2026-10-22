set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
exec 9> "$B/status/robotwin-close-controller.lock"
flock -n 9 || exit 0
pid=$(cat "$B/status/hf-tail-upload.pid")
while kill -0 "$pid" 2>/dev/null; do sleep 15; done
"$B/flux-action/.venv/bin/python" "$B/scripts/cleanup_robotwin_after_hf.py"
