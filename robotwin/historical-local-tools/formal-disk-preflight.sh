set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
{ date -u; df -BG "$S" "$B"; find "$S/runs" -maxdepth 2 -type f -name metrics.jsonl -o -name TRAIN_COMPLETE; } > "$B/status/formal-disk-preflight.txt"
