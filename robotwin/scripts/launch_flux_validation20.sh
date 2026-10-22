#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
cd "$B/RoboTwin"
exec 9>"$B/status/flux-validation.lock"
flock -n 9 || exit 0
source "$B/scripts/sim-env.sh"
export PATH="$B/sim-venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$PYTHONPATH" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$B/flux-action/.venv/bin/python" -c 'import torch;assert torch.cuda.mem_get_info()[0]>32*2**30'
cleanup() { if [ -n "${SERVER:-}" ]; then kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; fi; }
trap cleanup EXIT
for TASK in click_bell handover_block; do
 test -s "$B/status/flux-debug-${TASK}-ready.txt"
 export FLUX_EVAL_OUTPUT="$B/evaluation/$TASK/flux3000-validation20"
 export FLUX_EVAL_LOG="$B/status/flux-${TASK}-validation20.log"
 if [ -s "$FLUX_EVAL_OUTPUT/result.json" ]; then continue; fi
 PORT=$(bash "$B/XPolicyLab/utils/get_free_port.sh")
 bash "$B/XPolicyLab/policy/FLUX_Action/setup_eval_policy_server.sh" RoboTwin "$TASK" "$S/runs/$TASK/export-3000-ema" aloha_agilex joint 42 0 "$B/flux-action/.venv" "$PORT" > "$B/status/flux-${TASK}-validation-server.log" 2>&1 &
 SERVER=$!
 bash "$B/XPolicyLab/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER" "FLUX-$TASK-validation" 1200
 "$B/sim-venv/bin/python" "$B/scripts/run_eval_recorded.py" --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex --policy_name FLUX_Action --host localhost --port "$PORT" --protocol ws --eval_batch false --root_dir "$B/RoboTwin" --device_id 0 --additional_info 'ckpt_setting=flux3000-validation20,action_type=joint' --seed 8 --task_config demo_clean --test_num 20 --instruction_type seen --expert_check true --frequency 15 > "$FLUX_EVAL_LOG" 2>&1
 test -s "$FLUX_EVAL_OUTPUT/result.json"
 cleanup
 SERVER=
done
printf 'BOTH_VALIDATION20_COMPLETE\n' > "$B/status/flux-validation20-complete.txt"
