set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
nohup bash -c '"$1/policy-venv/bin/python" "$1/scripts/analyze_perf_traces.py" && "$1/policy-venv/bin/python" "$1/scripts/probe_flash_repeatability.py"' _ "$R" > "$R/status/perf-analysis.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$R/status/perf-analysis.pid"
