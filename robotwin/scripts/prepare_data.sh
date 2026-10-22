#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
PY="$B/flux-action/.venv/bin/python"
test -s status/runtime-ready.txt
test -f assets/robotwin-data.manifest.json
tar -xzf assets/xpolicy-utils.tar.gz -C "$B"
flock "$B/status/conversion.lock" "$PY" scripts/convert_robotwin.py
exec 8>"$B/status/index.lock"
flock 8
"$PY" scripts/make_configs.py
for task in click_bell handover_block; do
 "$PY" -m flux_action.cli index-lerobot --source-root "$B/assets/lerobot/$task" --output-dir "$B/assets/index/$task" --camera head=observation.images.cam_head --camera left_wrist=observation.images.cam_left_wrist --camera right_wrist=observation.images.cam_right_wrist --action-parameterization absolute --chunk-size 32 --val-episodes 0 --dataset-id "TianxingChen/RoboTwin2.0/$task/demo_clean" --revision 3dc3b798668feb99ac61cc9086d84cbcc3d79186 --hash-files
done
"$PY" scripts/verify_windows.py
"$PY" -c 'from flux_action.training.trainer import TrainConfig; from flux_action.config import PolicyConfig; import pathlib; b=pathlib.Path.cwd(); [PolicyConfig(**TrainConfig.from_file(b/"configs"/(t+".json")).policy).validate_training() for t in ["click_bell","handover_block"]]; print("CONFIG_AND_DATA_READY")' > status/data-ready.tmp
mv status/data-ready.tmp status/data-ready.txt
