set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,os,signal
p=pathlib.Path('/proc/3524167/cmdline')
if p.exists():
 assert b'/flux_piperx_20261008/scripts/inspect_robodojo_v3.py' in p.read_bytes()
 os.kill(3524167,signal.SIGTERM)
PY
sleep 2
cp "$R/status/v3-inspection.log" "$R/status/v3-inspection-attempt1.log"
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/inspect_robodojo_v3.py" > "$R/status/v3-inspection.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/v3-inspection.pid"