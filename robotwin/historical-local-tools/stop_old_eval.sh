B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/stop_old_eval.py" > "$B/status/stop-old-eval.log" 2>&1 < /dev/null &
