R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup bash "$R/scripts/install_robodojo_sim.sh" > "$R/status/sim-install.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/sim-install.pid"