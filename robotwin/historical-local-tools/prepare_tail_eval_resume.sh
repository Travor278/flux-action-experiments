B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup "$B/flux-action/.venv/bin/python" "$B/scripts/prepare_tail_eval_resume.py" > "$B/status/tail-eval-resume-preflight.log" 2>&1 < /dev/null &
