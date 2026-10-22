set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
{
 date -Is
 "$B/sim-venv/bin/python" --version
 "$B/flux-action/.venv/bin/python" --version
 command -v uv || true
 command -v nvcc || true
 nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
 df -h "$R"
 ldd --version | head -n 1
 cat /etc/os-release | head -n 5
} > "$R/status/runtime-preflight.txt" 2>&1