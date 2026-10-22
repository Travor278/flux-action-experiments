B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
sed -i 's/\r$//' "$B/scripts/repair_tail_verifier.sh"
nohup bash "$B/scripts/repair_tail_verifier.sh" > "$B/status/tail-verifier-repair.log" 2>&1 < /dev/null &
