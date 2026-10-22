set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
P="$B/flux-action-tail-v1"
test ! -e "$P"
mkdir "$P"
git -C "$B/flux-action" archive HEAD | tar -x -C "$P"
cp "$B/flux-action/src/flux_action/training/checkpoint.py" "$P/src/flux_action/training/checkpoint.py"
export PYTHONPATH="$P/src:$B/scripts:$P/tests" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="$B/flux-action/.venv/bin/python"
"$PY" "$B/scripts/install_tail_loss_patch.py" "$P"
"$PY" -m pytest -q "$B/scripts/test_tail_repair.py" "$P/tests/test_lerobot.py" "$P/tests/training/test_embodiment_plug.py" > "$B/status/tail-repair-tests.log" 2>&1
"$PY" "$B/scripts/prepare_tail_repair.py" > "$B/status/tail-repair-prepare.log" 2>&1
date -Is > "$B/status/tail-repair-cpu-complete.txt"
