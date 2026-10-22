set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 "$B/scripts/cleanup_two_gpu_diagnostic.py" > "$B/status/two-gpu-success-cleanup.log" 2>&1
