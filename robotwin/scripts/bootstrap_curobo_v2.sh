#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
exec 9>"$B/status/curobo-setup.lock"
flock -n 9 || exit 0
for i in $(seq 1 720); do
 if [ -s "$B/status/sim-dependencies-ready.txt" ]; then break; fi
 sleep 30
done
test -s "$B/status/sim-dependencies-ready.txt"
export UV_CACHE_DIR=/tmp/flux-piperx-sim-uv
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$B/sim-venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='8.9;9.0' MAX_JOBS=4
cd "$B/RoboTwin/envs"
if [ ! -d curobo/.git ]; then git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git; fi
git -C curobo rev-parse HEAD > "$B/status/curobo-revision.txt"
uv pip install --python "$B/sim-venv/bin/python" setuptools==69.5.1 wheel ninja 'warp-lang==1.12.0'
uv pip install --python "$B/sim-venv/bin/python" -e curobo --no-build-isolation
"$B/sim-venv/bin/python" - <<'PY' > "$B/status/curobo-import-check.tmp"
from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig
import torch
print('CUROBO_IMPORTS_OK',torch.__version__)
PY
mv "$B/status/curobo-import-check.tmp" "$B/status/curobo-import-check.txt"
