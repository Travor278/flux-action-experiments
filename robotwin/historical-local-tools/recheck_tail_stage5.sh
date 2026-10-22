B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
PYTHONPATH="$B/flux-action-tail-v1/src:$B/scripts" "$B/flux-action/.venv/bin/python" "$B/scripts/verify_tail_stage.py" handover_block 5 > "$B/status/handover-stage5-recheck.log" 2>&1
printf '%s\n' "$?" > "$B/status/handover-stage5-recheck-exit.txt"
