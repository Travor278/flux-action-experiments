#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
export UV_CACHE_DIR=/tmp/flux-piperx-uv-cache
export UV_PYTHON_INSTALL_DIR="$B/python"
if [ ! -d flux-action/.git ]; then git clone https://github.com/black-forest-labs/flux-action.git; fi
git -C flux-action checkout 830f1ce178d7aea52f18fde31d8e8e36ce6bae21
uv sync --project flux-action --locked --extra encoders --extra data
uv pip install --python "$B/flux-action/.venv/bin/python" 'natten==0.21.6+torch2100cu128' -f https://whl.natten.org/
uv pip install --python "$B/flux-action/.venv/bin/python" h5py opencv-python-headless 'numpy>=2,<2.3' pyyaml
"$B/flux-action/.venv/bin/python" -c 'import torch,natten,transformers,huggingface_hub; print(torch.__version__,natten.__version__,transformers.__version__,huggingface_hub.__version__)' > status/runtime-ready.txt
