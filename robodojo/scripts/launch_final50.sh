#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
TASK=$1
case "$TASK" in cover_blocks|match_and_pick_from_conveyor);; *) exit 2;; esac
OUT="$R/evaluation/$TASK/flux-10k-prefix-v1-final50"
if [ -f "$OUT/COMPLETE.json" ]; then
 "$R/policy-venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1]));assert r["verified"] and r["total"]==50' "$OUT/COMPLETE.json"
 exit 0
fi
exec 8> "$R/status/final50-$TASK.lock"
flock -n 8 || exit 3
export FLUX_ALLOW_RESUME=0
if [ "${2:-}" = '--resume-verified' ]; then export FLUX_ALLOW_RESUME=1; fi
RUN="$R/status/final50-jobs/$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)-$TASK"
mkdir -p "$RUN"
export FLUX_SCENE_RUN="$RUN" FLUX_EVAL_TASK="$TASK"
exec > >(tee -a "$RUN/launch.log") 2>&1
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$R/policy-venv/bin/python" "$R/scripts/check_final50_model.py" "$TASK"
"$R/policy-venv/bin/python" - <<'PY'
import subprocess,json,os,time,socket
from pathlib import Path
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');run=Path(os.environ['FLUX_SCENE_RUN']);task=os.environ['FLUX_EVAL_TASK']
rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
assert len(rows)==1 and '4090' in rows[0]
driver=rows[0].split(',')[1].strip();assert 570<=int(driver.split('.')[0])<595
r={'time':time.time(),'hostname':socket.gethostname(),'task':task,'run':str(run),'driver':driver,'phase':'model_verified'}
(run/'preflight.json').write_text(json.dumps(r,indent=2));(R/f'status/final50-current-{task}.json').write_text(json.dumps(r,indent=2))
PY
unset CUDA_VISIBLE_DEVICES VK_DRIVER_FILES
if [ -s /etc/vulkan/icd.d/nvidia_icd.json ]; then export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; elif [ -s /usr/share/vulkan/icd.d/nvidia_icd.json ]; then export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; fi
export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="$R/xdg-cache" XDG_CONFIG_HOME="$R/xdg-config" XDG_DATA_HOME="$R/xdg-data"
export FLUX_WS_PORT
FLUX_WS_PORT=$("$R/policy-venv/bin/python" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
server_pid='';mon=''
finish() {
 rc=$?
 if [ -n "$server_pid" ]; then kill "$server_pid" 2>/dev/null || true; fi
 if [ -n "$mon" ]; then kill "$mon" 2>/dev/null || true; fi
 if [ "$rc" -eq 0 ] && [ ! -f "$OUT/COMPLETE.json" ]; then rc=90; fi
 export FINAL_EXIT_CODE="$rc"
 "$R/policy-venv/bin/python" - <<'PY'
import os,json,time
from pathlib import Path
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');run=Path(os.environ['FLUX_SCENE_RUN']);task=os.environ['FLUX_EVAL_TASK']
r=json.loads((run/'preflight.json').read_text());r.update(finished=time.time(),exit_code=int(os.environ['FINAL_EXIT_CODE']),phase='complete' if int(os.environ['FINAL_EXIT_CODE'])==0 else 'failed')
(run/'result.json').write_text(json.dumps(r,indent=2));(R/f'status/final50-current-{task}.json').write_text(json.dumps(r,indent=2))
PY
 exit "$rc"
}
trap finish EXIT
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.used,utilization.gpu --format=csv -l 5 > "$RUN/gpu.csv" &
mon=$!
export PYTHONPATH="$R/ws-deps:$R/scripts:$R/flux-action/src:$R/XPolicyLab:$R"
"$R/policy-venv/bin/python" "$R/scripts/serve_final50.py" "$TASK" > "$RUN/server.log" 2>&1 &
server_pid=$!
for i in $(seq 1 300); do
 if [ -f "$RUN/server-ready.json" ]; then break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then exit 91; fi
 sleep 2
done
test -f "$RUN/server-ready.json"
export PYTHONPATH="$R/RoboDojo:$R/XPolicyLab:$R:$R/scripts"
cd "$R/RoboDojo"
"$R/sim-venv/bin/python" "$R/scripts/run_final50.py" "$TASK" > "$RUN/client.log" 2>&1
"$R/policy-venv/bin/python" "$R/scripts/audit_final50.py" "$TASK" > "$RUN/audit.log" 2>&1
