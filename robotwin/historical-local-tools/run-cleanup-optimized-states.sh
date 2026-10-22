set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" "$B/scripts/cleanup_optimized_training_states.py" > "$B/status/optimized-training-state-cleanup.log" 2>&1
