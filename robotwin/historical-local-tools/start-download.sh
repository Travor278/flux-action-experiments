set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
export UV_CACHE_DIR=/tmp/flux-piperx-download-uv
export HF_XET_CACHE=/tmp/flux-piperx-xet
nohup uv run --no-project --python /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3 --with huggingface-hub --with safetensors --with numpy python "$B/scripts/download.py" > "$B/status/download.log" 2>&1 < /dev/null &
echo $! > "$B/status/download.pid"
