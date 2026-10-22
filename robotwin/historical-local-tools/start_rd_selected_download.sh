R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup bash "$R/scripts/start_selected_download_when_ready.sh" > "$R/status/selected-download.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/selected-download.pid"