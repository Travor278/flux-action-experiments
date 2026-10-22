set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
sed -i 's/\r$//' "$B/scripts/launch_flux_tail_test100.sh"
bash -n "$B/scripts/launch_flux_tail_test100.sh"
nohup bash "$B/scripts/launch_flux_tail_test100.sh" > "$B/status/flux-tail-final-test-launch.log" 2>&1 < /dev/null & echo $! > "$B/status/flux-tail-test.pid"
