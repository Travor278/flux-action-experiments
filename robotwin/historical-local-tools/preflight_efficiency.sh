set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
export PYTHONPATH="$B/flux-action/src:$B/scripts" OMP_NUM_THREADS=4
bash -n scripts/launch_efficiency_sweep.sh
"$B/flux-action/.venv/bin/python" -m py_compile scripts/async_cpu_ema.py scripts/test_async_ema.py scripts/train_efficiency_worker.py scripts/run_efficiency_sweep.py
"$B/flux-action/.venv/bin/python" scripts/test_async_ema.py > status/async-ema-unit.log 2>&1
