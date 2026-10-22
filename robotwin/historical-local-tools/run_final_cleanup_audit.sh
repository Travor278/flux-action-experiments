B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
sed -i 's/\r$//' "$B/scripts/verify_failed_training_cleanup.sh"
nohup bash "$B/scripts/verify_failed_training_cleanup.sh" > "$B/status/final-cleanup-audit.log" 2>&1 < /dev/null &
