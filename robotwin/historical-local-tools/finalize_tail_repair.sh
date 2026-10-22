set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
sed -i 's/\r$//' "$B/scripts/launch_train_tail_repaired.sh"
bash -n "$B/scripts/launch_train_tail_repaired.sh"
PYTHONPATH="$B/flux-action-tail-v1/src:$B/scripts" "$B/flux-action/.venv/bin/python" "$B/scripts/finalize_tail_repair.py" > "$B/status/tail-repair-finalize.log" 2>&1
