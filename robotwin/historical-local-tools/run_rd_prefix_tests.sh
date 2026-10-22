set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/scripts"
OMP_NUM_THREADS=2 "$R/policy-venv/bin/python" -m unittest test_prefix_memory test_video_boundary test_history_reader test_history_dataset > "$R/status/prefix-memory-tests.log" 2>&1
printf PASS > "$R/status/prefix-memory-tests-passed.txt"