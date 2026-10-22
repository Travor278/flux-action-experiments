set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
bash -n "$B/scripts/launch_train_optimized.sh"
"$B/flux-action/.venv/bin/python" -m py_compile "$B/scripts/prepare_optimized_10k.py" "$B/scripts/verify_optimized_export.py"
printf 'OPTIMIZED_LAUNCH_SYNTAX_PASS\n' > "$B/status/optimized-launch-syntax.txt"
