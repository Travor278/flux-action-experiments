B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup bash "$B/scripts/install_rpc_deps.sh" > "$B/status/rpc-deps.log" 2>&1 < /dev/null &
echo $! > "$B/status/rpc-deps.pid"
