set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
exec 8>"$B/status/index.lock"
flock -n 8 || exit 0
PY="$B/flux-action/.venv/bin/python"
"$PY" scripts/make_configs.py
for task in click_bell handover_block; do
 test -f "$B/assets/lerobot/$task/CONVERSION_COMPLETE.json"
 "$PY" -m flux_action.cli index-lerobot --source-root "$B/assets/lerobot/$task" --output-dir "$B/assets/index/$task" --camera head=observation.images.cam_head --camera left_wrist=observation.images.cam_left_wrist --camera right_wrist=observation.images.cam_right_wrist --action-parameterization absolute --chunk-size 32 --val-episodes 0 --dataset-id "TianxingChen/RoboTwin2.0/$task/demo_clean" --revision 3dc3b798668feb99ac61cc9086d84cbcc3d79186 --hash-files
done
"$PY" scripts/verify_windows.py
