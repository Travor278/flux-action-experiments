B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cp "$B/status/flux-debug-ready.txt" "$B/status/flux-debug-click_bell-ready.txt"
nohup bash "$B/scripts/debug_flux_rpc_v2.sh" handover_block > "$B/status/flux-debug-handover_block-launch.log" 2>&1 < /dev/null &
echo $! > "$B/status/flux-debug-handover_block-launch.pid"
