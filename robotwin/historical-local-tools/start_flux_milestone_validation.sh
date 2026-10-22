set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
bash -n "$B/scripts/launch_flux_milestone_validation.sh"
test -s "$B/status/optimized-final-artifacts-verified.json"
nohup bash "$B/scripts/launch_flux_milestone_validation.sh" > "$B/status/flux-milestone-validation-launch.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/flux-milestone-validation.pid"
