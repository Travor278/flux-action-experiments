#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=${B/\/hdd\//\/ssd\/}
cd "$B/RoboTwin"
exec 9>"$B/status/flux-final-test.lock"
flock -n 9 || exit 3
exec 8>"$B/status/flux-milestone-validation.lock"
flock -n 8 || exit 4
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,torch
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
p=json.loads((B/'status/tail-handover-resume-plan.json').read_text());assert p['retained_episodes']==33 and p['retained_successes']==9
assert not (B/'evaluation/handover_block/flux-tail-10k-test100/result.json').exists()
assert json.loads((B/'status/tail-final-artifacts-verified.json').read_text())['complete']
assert torch.cuda.mem_get_info()[0]>32*2**30
PY
source "$B/scripts/sim-env.sh"
export PATH="$B/sim-venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$PYTHONPATH" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK=handover_block
LABEL=flux-tail-10k-test100
CHECKPOINT="$S/runs/$TASK-10k-tail-v1/export-10000-ema"
export FLUX_EVAL_OUTPUT="$B/evaluation/$TASK/$LABEL" FLUX_EVAL_LOG="$B/status/$LABEL-$TASK.log"
PORT=$(bash "$B/XPolicyLab/utils/get_free_port.sh")
bash "$B/XPolicyLab/policy/FLUX_Action/setup_eval_policy_server.sh" RoboTwin "$TASK" "$CHECKPOINT" aloha_agilex joint 42 0 "$B/flux-action/.venv" "$PORT" >> "$B/status/$LABEL-$TASK-server.log" 2>&1 &
SERVER=$!
cleanup(){ kill "$SERVER" 2>/dev/null || true;wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT
printf '10000 %s %s %s\n' "$TASK" "$SERVER" "$PORT" > "$B/status/flux-tail-test-current.txt"
bash "$B/XPolicyLab/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER" "FLUX-tail-resume33" 1200
"$B/sim-venv/bin/python" "$B/scripts/run_eval_tail_resume.py" --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex --policy_name FLUX_Action --host localhost --port "$PORT" --protocol ws --eval_batch false --root_dir "$B/RoboTwin" --device_id 0 --additional_info "ckpt_setting=$LABEL,action_type=joint" --seed 0 --task_config demo_clean --test_num 100 --instruction_type seen --expert_check true --frequency 15 >> "$FLUX_EVAL_LOG" 2>&1
"$B/flux-action/.venv/bin/python" "$B/scripts/audit_tail_final100.py" handover_block > "$B/status/flux-tail-handover-final-audit.log" 2>&1
printf 'BOTH_TAIL_10K_TEST100_COMPLETE\n' > "$B/status/flux-tail-test-complete.txt"
