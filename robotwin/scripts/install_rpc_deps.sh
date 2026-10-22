#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
export UV_CACHE_DIR=/tmp/flux-piperx-uv-cache
for ENV in "$B/flux-action/.venv" "$B/sim-venv"; do
 NAME=$(basename "$(dirname "$ENV")")-$(basename "$ENV")
 "$ENV/bin/python" -c 'import importlib.metadata as m;print("\n".join(n+"=="+m.version(n) for n in ["torch","torchvision","numpy"]))' > "$B/status/rpc-$NAME-constraints.txt"
 uv pip install --python "$ENV/bin/python" -c "$B/status/rpc-$NAME-constraints.txt" 'websockets>=14,<16' msgpack msgpack-numpy 'pydantic>=2.5'
 "$ENV/bin/python" -c 'import websockets,msgpack,msgpack_numpy,pydantic,cv2,yaml;print("RPC_IMPORTS_OK",websockets.__version__,pydantic.__version__)'
done
