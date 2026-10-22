set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cp "$R/status/video-sample-audit.log" "$R/status/video-sample-audit-first-span-assertion.log"
nohup bash -c 'flock -n "$1/status/video-sample-audit.lock" "$1/policy-venv/bin/python" "$1/scripts/audit_video_samples.py"; code=$?; echo "$code" > "$1/status/video-sample-audit.exitcode"' _ "$R" > "$R/status/video-sample-audit.log" 2>&1 < /dev/null &
echo $! > "$R/status/video-sample-audit.pid"