B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup bash "$B/scripts/prepare_robodojo_remote.sh" > "$B/status/robodojo-prepare.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/robodojo-prepare.pid"