#!/usr/bin/env bash
set -euo pipefail
W=/home/USER/rpent-data/codex_remote_ops/flux_env_compat_20261010
E=/home/USER/rpent-data/venvs/travor-train-flux-cu128
UV=/home/USER/.local/bin/uv
mkdir -p "$W/status" "$W/cache"
export UV_CACHE_DIR="$W/cache" UV_HTTP_TIMEOUT=120 UV_PYTHON_INSTALL_DIR="$W/python"
[ ! -e "$E/pyvenv.cfg" ] || { echo 'Environment already exists; inspect before reuse'; exit 3; }
"$UV" python install 3.12
"$UV" venv --python 3.12 --prompt travor-train-flux-cu128 "$E"
cat > "$W/requirements.in" <<'REQ'
lerobot[training]==0.6.1
transformers==5.16.1
torch==2.10.0+cu128
torchvision==0.25.0+cu128
torchcodec==0.10.0
numpy>=2.0,<2.3
pyarrow==21.0.0
av==15.1.0
einops==0.8.1
safetensors>=0.4.3,<1
msgpack>=1.0,<2
websockets>=13,<16
pytest>=8,<10
REQ
"$UV" pip install --python "$E/bin/python" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 -r "$W/requirements.in"
"$UV" pip check --python "$E/bin/python"
"$UV" pip freeze --python "$E/bin/python" > "$W/status/installed.lock"
date -Is > "$W/status/packages-complete.txt"
