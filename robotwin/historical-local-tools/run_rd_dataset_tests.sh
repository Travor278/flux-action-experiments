set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/scripts"
OMP_NUM_THREADS=2 "$R/policy-venv/bin/python" -m unittest test_history_dataset > "$R/status/history-dataset-tests.log" 2>&1
"$R/policy-venv/bin/python" build_history_preflight_index.py > "$R/status/history-index-build.log" 2>&1
printf PASS > "$R/status/history-dataset-index-tests-passed.txt"