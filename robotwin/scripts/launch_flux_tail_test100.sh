#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=${B/\/hdd\//\/ssd\/}
cd "$B/RoboTwin"
exec 9>"$B/status/flux-final-test.lock"
flock -n 9 || exit 3
exec 8>"$B/status/flux-milestone-validation.lock"
flock -n 8 || exit 4
test -s "$B/status/old-eval-user-cancelled.json"
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
assert json.loads((B/'status/old-eval-user-cancelled.json').read_text())['stopped_verified']
audit=json.loads((B/'status/tail-final-artifacts-verified.json').read_text());assert audit['complete']
plan={'time':time.time(),'selection_rule':'User explicitly requested 10000-step models only; no validation-based checkpoint selection','step':10000,'test_num':100,'seed_parameter':0,'expert_check':True,'tasks':{}}
for task in ['click_bell','handover_block']:
 assert audit['tasks'][task]['complete'] and audit['tasks'][task]['step']==10000
 cp=S/f'runs/{task}-10k-tail-v1/export-10000-ema'
 assert all((cp/n).is_file() for n in ['model.safetensors','config.json','manifest.json'])
 plan['tasks'][task]={'checkpoint':str(cp),'output':str(B/f'evaluation/{task}/flux-tail-10k-test100')}
p=B/'status/flux-tail-final-test-plan.json'
if p.exists():assert json.loads(p.read_text())['tasks']==plan['tasks']
else:p.write_text(json.dumps(plan,indent=2))
PY
source "$B/scripts/sim-env.sh"
export PATH="$B/sim-venv/bin:$PATH" PYTHONPATH="$B/flux-action/src:$PYTHONPATH" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cleanup() { if [ -n "${SERVER:-}" ]; then kill "$SERVER" 2>/dev/null || true;wait "$SERVER" 2>/dev/null || true;fi; }
trap cleanup EXIT
for TASK in click_bell handover_block; do
 STEP=10000
 CHECKPOINT="$S/runs/$TASK-10k-tail-v1/export-10000-ema"
 LABEL=flux-tail-10k-test100
 export FLUX_EVAL_OUTPUT="$B/evaluation/$TASK/$LABEL"
 export FLUX_EVAL_LOG="$B/status/$LABEL-$TASK.log"
 if test -s "$FLUX_EVAL_OUTPUT/result.json"; then
  python -c 'import json,sys;r=json.load(open(sys.argv[1]));assert r["complete"] and r["episodes"]==100 and r["split"]=="test"' "$FLUX_EVAL_OUTPUT/result.json"
  continue
 fi
 if test -e "$FLUX_EVAL_OUTPUT/cases.jsonl"; then echo 'Existing partial test: diagnose and resume explicitly';exit 5;fi
 "$B/flux-action/.venv/bin/python" -c 'import torch;assert torch.cuda.mem_get_info()[0]>32*2**30'
 PORT=$(bash "$B/XPolicyLab/utils/get_free_port.sh")
 bash "$B/XPolicyLab/policy/FLUX_Action/setup_eval_policy_server.sh" RoboTwin "$TASK" "$CHECKPOINT" aloha_agilex joint 42 0 "$B/flux-action/.venv" "$PORT" > "$B/status/$LABEL-$TASK-server.log" 2>&1 &
 SERVER=$!
 printf '%s %s %s %s\n' "$STEP" "$TASK" "$SERVER" "$PORT" > "$B/status/flux-tail-test-current.txt"
 bash "$B/XPolicyLab/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER" "FLUX-10k-final-$TASK" 1200
 for ENCODED in false true; do
  "$B/sim-venv/bin/python" "$B/XPolicyLab/utils/debug_env_client.py" --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex --policy_name FLUX_Action --protocol ws --host localhost --port "$PORT" --eval_batch false --eval_episode_num 1 --obs_encoded "$ENCODED" > "$B/status/$LABEL-$TASK-debug-$ENCODED.log" 2>&1
 done
 "$B/sim-venv/bin/python" "$B/scripts/run_eval_paired.py" --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex --policy_name FLUX_Action --host localhost --port "$PORT" --protocol ws --eval_batch false --root_dir "$B/RoboTwin" --device_id 0 --additional_info "ckpt_setting=$LABEL,action_type=joint" --seed 0 --task_config demo_clean --test_num 100 --instruction_type seen --expert_check true --frequency 15 > "$FLUX_EVAL_LOG" 2>&1
 test -s "$FLUX_EVAL_OUTPUT/result.json"
 cleanup;SERVER=
done
printf 'BOTH_TAIL_10K_TEST100_COMPLETE\n' > "$B/status/flux-tail-test-complete.txt"
