set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
test -s "$B/status/curated-replay-preflight.json"
nohup bash "$B/scripts/launch_flux_milestone_validation_v2.sh" >> "$B/status/flux-milestone-validation-launch-v2.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/flux-milestone-validation.pid"
