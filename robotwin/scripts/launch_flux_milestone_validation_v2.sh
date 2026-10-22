#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
cd "$B/RoboTwin"
exec 9>"$B/status/flux-milestone-validation.lock"
flock -n 9 || exit 0
test -s "$B/status/optimized-final-artifacts-verified.json"
source "$B/scripts/sim-env.sh"
export PATH="$B/sim-venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$PYTHONPATH" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cleanup() { if [ -n "${SERVER:-}" ]; then kill "$SERVER" 2>/dev/null || true;wait "$SERVER" 2>/dev/null || true;fi; }
trap cleanup EXIT
for STEP in 10000 5000 3000 1000; do
 for TASK in click_bell handover_block; do
  LABEL="flux-opt-${STEP}-validation20"
  export FLUX_EVAL_OUTPUT="$B/evaluation/$TASK/$LABEL"
  export FLUX_EVAL_LOG="$B/status/$LABEL-$TASK.log"
  if test -s "$FLUX_EVAL_OUTPUT/result.json"; then continue;fi
  "$B/flux-action/.venv/bin/python" -c 'import torch;assert torch.cuda.mem_get_info()[0]>32*2**30'
  PORT=$(bash "$B/XPolicyLab/utils/get_free_port.sh")
  bash "$B/XPolicyLab/policy/FLUX_Action/setup_eval_policy_server.sh" RoboTwin "$TASK" "$S/runs/$TASK-10k-opt-v1/export-$STEP-ema" aloha_agilex joint 42 0 "$B/flux-action/.venv" "$PORT" > "$B/status/$LABEL-$TASK-server.log" 2>&1 &
  SERVER=$!
  printf '%s %s %s %s\n' "$STEP" "$TASK" "$SERVER" "$PORT" > "$B/status/flux-milestone-validation-current.txt"
  bash "$B/XPolicyLab/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER" "FLUX-$TASK-$STEP" 1200
  for ENCODED in false true; do
   "$B/sim-venv/bin/python" "$B/XPolicyLab/utils/debug_env_client.py" --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex --policy_name FLUX_Action --protocol ws --host localhost --port "$PORT" --eval_batch false --eval_episode_num 1 --obs_encoded "$ENCODED" > "$B/status/$LABEL-$TASK-debug-$ENCODED.log" 2>&1
  done
  "$B/sim-venv/bin/python" "$B/scripts/run_eval_curated.py" --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex --policy_name FLUX_Action --host localhost --port "$PORT" --protocol ws --eval_batch false --root_dir "$B/RoboTwin" --device_id 0 --additional_info "ckpt_setting=$LABEL,action_type=joint" --seed 8 --task_config demo_clean --test_num 20 --instruction_type seen --expert_check true --frequency 15 >> "$FLUX_EVAL_LOG" 2>&1
  test -s "$FLUX_EVAL_OUTPUT/result.json"
  cleanup;SERVER=
 done
done
printf 'ALL_8_MILESTONE_VALIDATIONS_COMPLETE\n' > "$B/status/flux-milestone-validation-complete.txt"
