#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B/RoboTwin"
exec 9>"$B/status/flux-final-test.lock"
flock -n 9 || exit 3
# Final tests must not overlap the validation simulator/policy pipeline.
exec 8>"$B/status/flux-milestone-validation.lock"
flock -n 8 || exit 4
test -s "$B/status/flux-milestone-validation-complete.txt"
"$B/flux-action/.venv/bin/python" "$B/scripts/select_flux_checkpoint.py" > "$B/status/flux-checkpoint-selection.log" 2>&1
source "$B/scripts/sim-env.sh"
export PATH="$B/sim-venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$PYTHONPATH" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cleanup() { if [ -n "${SERVER:-}" ]; then kill "$SERVER" 2>/dev/null || true;wait "$SERVER" 2>/dev/null || true;fi; }
trap cleanup EXIT
for TASK in click_bell handover_block; do
 read -r STEP CHECKPOINT < <(python -c 'import json,sys;r=json.load(open(sys.argv[1]))["tasks"][sys.argv[2]];print(r["selected"]["step"],r["checkpoint"])' "$B/status/flux-checkpoint-selection.json" "$TASK")
 LABEL="flux-selected-${STEP}-test100"
 export FLUX_EVAL_OUTPUT="$B/evaluation/$TASK/$LABEL"
 export FLUX_EVAL_LOG="$B/status/$LABEL-$TASK.log"
 if test -s "$FLUX_EVAL_OUTPUT/result.json"; then continue;fi
 # Never silently repeat an incomplete held-out test after a failure.
 if test -e "$FLUX_EVAL_OUTPUT/cases.jsonl";then echo 'Existing partial final test; inspect and resume explicitly';exit 5;fi
 "$B/flux-action/.venv/bin/python" -c 'import torch;assert torch.cuda.mem_get_info()[0]>32*2**30'
 PORT=$(bash "$B/XPolicyLab/utils/get_free_port.sh")
 bash "$B/XPolicyLab/policy/FLUX_Action/setup_eval_policy_server.sh" RoboTwin "$TASK" "$CHECKPOINT" aloha_agilex joint 42 0 "$B/flux-action/.venv" "$PORT" > "$B/status/$LABEL-$TASK-server.log" 2>&1 &
 SERVER=$!
 printf '%s %s %s %s\n' "$STEP" "$TASK" "$SERVER" "$PORT" > "$B/status/flux-final-test-current.txt"
 bash "$B/XPolicyLab/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER" "FLUX-final-$TASK" 1200
 "$B/sim-venv/bin/python" "$B/scripts/run_eval_paired.py" --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex --policy_name FLUX_Action --host localhost --port "$PORT" --protocol ws --eval_batch false --root_dir "$B/RoboTwin" --device_id 0 --additional_info "ckpt_setting=$LABEL,action_type=joint" --seed 0 --task_config demo_clean --test_num 100 --instruction_type seen --expert_check true --frequency 15 > "$FLUX_EVAL_LOG" 2>&1
 test -s "$FLUX_EVAL_OUTPUT/result.json"
 cleanup;SERVER=
done
printf 'BOTH_SELECTED_TEST100_COMPLETE\n' > "$B/status/flux-final-test-complete.txt"
