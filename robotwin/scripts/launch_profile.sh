#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
export PATH="$B/flux-action/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$B/flux-action/src:$B:${PYTHONPATH:-}"
test -s status/runtime-ready.txt
test -s status/data-ready.txt
test -s status/windows-verified.json
test -f assets/flux-base.manifest.json
/opt/a2r/venv/bin/python - <<'PY' > status/pi05-image-inspection.txt 2>&1 || true
import importlib.metadata,inspect
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
for p in ['torch','lerobot','transformers','accelerate']:
 print(p,importlib.metadata.version(p))
print(inspect.getsource(LeRobotDatasetMetadata.__init__))
PY
if [ ! -s status/pi05-data-verified.json ]; then
 /opt/a2r/venv/bin/python scripts/convert_pi05_v30.py > status/pi05-v30-conversion.log 2>&1 || echo 'PI05 dataset conversion requires follow-up; FLUX profiling remains independent'
fi
python -c 'import torch; assert torch.cuda.device_count()==8; print(torch.__version__,torch.cuda.get_device_name(),flush=True)'
python scripts/benchmark_scaling.py
