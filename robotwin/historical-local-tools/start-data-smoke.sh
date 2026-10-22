set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
tar -xzf "$B/assets/xpolicy-utils.tar.gz" -C "$B"
nohup env UV_CACHE_DIR=/tmp/flux-piperx-uv-cache uv run --no-project --python /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3 --with 'numpy>=2,<2.3' --with 'av>=14,<16' --with h5py --with 'pyarrow>=19,<22' --with opencv-python-headless --with pyyaml python "$B/scripts/data_smoke.py" > "$B/status/data-smoke.log" 2>&1 < /dev/null &
