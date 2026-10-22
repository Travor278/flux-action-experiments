set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup env UV_CACHE_DIR=/tmp/flux-piperx-download-uv uv run --no-project --python /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3 --with h5py python "$B/scripts/inspect_data.py" > "$B/status/inspect.log" 2>&1 < /dev/null &
