set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/scripts"
"$R/policy-venv/bin/python" install_temporal_masks.py > "$R/status/temporal-mask-install.log" 2>&1
"$R/policy-venv/bin/python" -m unittest test_temporal_masks test_video_boundary test_history_reader > "$R/status/temporal-mask-tests.log" 2>&1
echo PASS > "$R/status/temporal-mask-cpu-tests-passed.txt"