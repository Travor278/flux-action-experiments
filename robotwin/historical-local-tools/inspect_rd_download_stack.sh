set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export UV_CACHE_DIR="$R/uv-cache"
uv pip install --python "$R/policy-venv/bin/python" py-spy > "$R/status/py-spy-install.log" 2>&1
"$R/policy-venv/bin/py-spy" dump --pid 17679 > "$R/status/download-python-stack.txt" 2>&1