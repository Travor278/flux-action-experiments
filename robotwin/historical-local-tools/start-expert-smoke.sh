set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
test -s "$B/status/curobo-import-check.txt"
source "$B/scripts/sim-env.sh"
cd "$B/RoboTwin"
ulimit -c 0
nohup timeout 600 "$B/sim-venv/bin/python" "$B/scripts/expert_smoke.py" > "$B/status/expert-smoke.log" 2>&1 < /dev/null &
echo $! > "$B/status/expert-smoke.pid"
