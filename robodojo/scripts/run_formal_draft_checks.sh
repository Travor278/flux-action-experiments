R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/scripts:$R/flux-action/src" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
"$R/policy-venv/bin/python" -m pytest -q "$R/scripts/test_training_contract.py" > "$R/status/training-contract-tests.log" 2>&1
code=$?
if [ "$code" -eq 0 ]; then "$R/policy-venv/bin/python" "$R/scripts/prepare_formal_drafts.py" > "$R/status/formal-drafts.log" 2>&1; fi
