set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/archive_robotwin_records.py" > "$B/status/robotwin-records.log" 2>&1 < /dev/null &