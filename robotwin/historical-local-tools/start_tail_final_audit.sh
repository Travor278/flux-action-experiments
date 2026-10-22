B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/verify_tail_final_artifacts.py" > "$B/status/tail-final-artifact-audit.log" 2>&1 < /dev/null & echo $! > "$B/status/tail-final-artifact-audit.pid"
