B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup bash "$B/scripts/finish_robotwin_close.sh" > "$B/status/robotwin-close-controller.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/robotwin-close-controller.pid"