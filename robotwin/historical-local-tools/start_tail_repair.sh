B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
sed -i 's/\r$//' "$B/scripts/install_tail_repair.sh"
nohup bash "$B/scripts/install_tail_repair.sh" > "$B/status/tail-repair-install.log" 2>&1 < /dev/null & echo $! > "$B/status/tail-repair-install.pid"
