R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup bash "$R/scripts/install_rd_sim_extensions.sh" > "$R/status/sim-extensions-install.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/sim-extensions-install.pid"