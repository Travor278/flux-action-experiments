set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
export PYTHONPATH="$B/flux-action/src:$B/scripts"
bash -n scripts/launch_train_optimized.sh
"$B/flux-action/.venv/bin/python" scripts/prepare_optimized_10k.py > status/prepare-optimized-ten-k.log 2>&1
