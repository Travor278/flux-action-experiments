B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup bash "$B/scripts/launch_flux_validation20.sh" > "$B/status/flux-validation20-launch.log" 2>&1 < /dev/null &
echo $! > "$B/status/flux-validation20.pid"
