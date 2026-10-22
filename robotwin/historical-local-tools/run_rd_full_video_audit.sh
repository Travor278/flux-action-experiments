set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup bash -c 'flock -n "$1/status/all-video-decode.lock" "$1/policy-venv/bin/python" "$1/scripts/audit_all_selected_videos.py"; code=$?; echo "$code" > "$1/status/all-video-decode.exitcode"' _ "$R" > "$R/status/all-video-decode.log" 2>&1 < /dev/null &
echo $! > "$R/status/all-video-decode.pid"