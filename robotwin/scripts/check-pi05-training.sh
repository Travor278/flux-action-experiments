set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
P=/workspace/hdd/codex_remote_ops/A2R_pi05_20261007/venv
export PYTHONPATH="$B/flux-action/src:$B/scripts:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE=/tmp/flux-piperx-pi05-datasets
"$P/bin/python" -c 'import runpy,os; scope=runpy.run_path(os.environ["PYTHONPATH"].split(":")[1]+"/test_matched_sampler.py"); [v() for k,v in scope.items() if k.startswith("test_")]; print("MATCHED_SAMPLER_2_TESTS_PASSED")'
for task in click_bell handover_block; do
 "$P/bin/python" "$B/scripts/train_pi05.py" "$task" --preflight-only
done
echo PI05_TRAINING_CONFIG_CHECK_PASSED
