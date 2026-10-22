#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=/workspace/ssd/codex_remote_ops/flux_piperx_20261008
TASK=${1:?task required}
case "$TASK" in click_bell|handover_block) ;; *) exit 2;; esac
cd "$B"
export PYTHONPATH="$B:$B/flux-action/src:$B/XPolicyLab" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PATH="$B/sim-venv/bin:$PATH"
test -s "$B/status/flux-final-artifacts-verified.json"
"$B/flux-action/.venv/bin/python" -c 'import torch,websockets,msgpack,msgpack_numpy,pydantic,cv2,yaml;assert torch.cuda.mem_get_info()[0]>32*2**30;print("RPC_GPU_PREFLIGHT_OK",flush=True)'
"$B/sim-venv/bin/python" -c 'import websockets,msgpack,msgpack_numpy,pydantic,cv2,yaml'
PORT=$(bash "$B/XPolicyLab/utils/get_free_port.sh")
printf '%s\n' "$PORT" > "$B/status/flux-debug-${TASK}-port.txt"
bash "$B/XPolicyLab/policy/FLUX_Action/setup_eval_policy_server.sh" RoboTwin "$TASK" "$S/runs/$TASK/export-3000-ema" aloha_agilex joint 42 0 "$B/flux-action/.venv" "$PORT" > "$B/status/flux-debug-${TASK}-server.log" 2>&1 &
SERVER=$!
echo "$SERVER" > "$B/status/flux-debug-${TASK}-server.pid"
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
bash "$B/XPolicyLab/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER" FLUX-debug 1200
for ENCODED in false true; do
 "$B/sim-venv/bin/python" "$B/XPolicyLab/utils/debug_env_client.py" --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex --policy_name FLUX_Action --protocol ws --host localhost --port "$PORT" --eval_batch false --eval_episode_num 1 --obs_encoded "$ENCODED" > "$B/status/flux-debug-${TASK}-${ENCODED}.log" 2>&1
done
printf 'PLAIN_AND_ENCODED_DEBUG_PASS\n' > "$B/status/flux-debug-${TASK}-ready.txt"
