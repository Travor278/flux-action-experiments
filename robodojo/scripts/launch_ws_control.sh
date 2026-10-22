#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
RUN="$R/status/ws-control-jobs/$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launch.log") 2>&1
export FLUX_SCENE_RUN="$RUN"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,subprocess,os,time,socket,hashlib
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(os.environ['FLUX_SCENE_RUN'])
q=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(q)==1,q
name,driver,memory=[v.strip() for v in q[0].split(',')];assert '4090' in name
assert json.loads((R/'status/selected-download-verified.json').read_text())['complete']
record={'time':time.time(),'hostname':socket.gethostname(),'gpu':name,'driver':driver,'memory_mib':int(memory),'run':str(out),'script_sha256':hashlib.sha256((R/'scripts/probe_ws_control.py').read_bytes()).hexdigest(),'phase':'driver_verified','ws_passed':False}
(out/'preflight.json').write_text(json.dumps(record,indent=2));(R/'status/ws-control-current.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
major=int(driver.split('.')[0])
if major<570 or major>=595:
 record['phase']='driver_rejected';record['reason']='Driver outside candidate range for IsaacSim5.1; no repeated RTX crash on known595 branch'
 (out/'result.json').write_text(json.dumps(record,indent=2));(R/'status/ws-control-current.json').write_text(json.dumps(record,indent=2));raise SystemExit(78)
PY
unset CUDA_VISIBLE_DEVICES VK_DRIVER_FILES
if [ -s /etc/vulkan/icd.d/nvidia_icd.json ]; then export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; elif [ -s /usr/share/vulkan/icd.d/nvidia_icd.json ]; then export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; fi
export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$R/xdg-cache" XDG_CONFIG_HOME="$R/xdg-config" XDG_DATA_HOME="$R/xdg-data"
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.used,utilization.gpu --format=csv -l 5 > "$RUN/gpu.csv" 2>&1 &
telemetry_pid=$!
server_pid=''
trap 'kill "$telemetry_pid" ${server_pid:-} 2>/dev/null || true' EXIT
export FLUX_WS_PORT=48691 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export PYTHONPATH="$R/ws-deps:$R/scripts:$R/flux-action/src:$R/XPolicyLab:$R"
"$R/policy-venv/bin/python" "$R/scripts/ws_probe_server.py" > "$RUN/server.log" 2>&1 &
server_pid=$!
for i in $(seq 1 180); do
 if [ -f "$RUN/server-ready.json" ]; then break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then exit 91; fi
 sleep 2
done
test -f "$RUN/server-ready.json"
set +e
export PYTHONPATH="$R/RoboDojo:$R/XPolicyLab:$R:$R/scripts" ROBODOJO_RUN_ID="scene-control-$(date +%s)"
cd "$R/RoboDojo"
code=0
for task in cover_blocks match_and_pick_from_conveyor; do
 timeout --signal=TERM --kill-after=15s 900s "$R/sim-venv/bin/python" "$R/scripts/probe_ws_control.py" "$task" > "$RUN/$task.log" 2>&1
 code=$?
 if [ "$code" -eq 0 ] && [ ! -f "$RUN/$task/ws-control-verified.json" ]; then code=90; fi
 if [ "$code" -ne 0 ]; then break; fi
done
set -e
export RENDER_EXIT_CODE="$code"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import os,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(os.environ['FLUX_SCENE_RUN']);r=json.loads((out/'preflight.json').read_text());code=int(os.environ['RENDER_EXIT_CODE']);p=out/'match_and_pick_from_conveyor/ws-control-verified.json'
r.update(finished=time.time(),exit_code=code,phase='ws_passed' if code==0 and p.exists() else 'ws_failed',ws_passed=code==0 and p.exists())
if r['ws_passed']:
 r['tasks']={task:json.loads((out/task/'ws-control-verified.json').read_text()) for task in ('cover_blocks','match_and_pick_from_conveyor')};assert all(v['verified'] and v['control_ticks']==53 for v in r['tasks'].values())
 events=[json.loads(line) for line in (out/'server-events.jsonl').read_text().splitlines()]
 actions=[e for e in events if e['event']=='action'];assert len(actions)==106
 assert [e['tick'] for e in actions]==list(range(53))*2
 assert sum(e['tick']==52 and e['anchors']==[0,25,50] for e in actions)==2
 assert events[-1]['event']=='reset'
 r['server_history_audit']=True
(out/'result.json').write_text(json.dumps(r,indent=2));(R/'status/ws-control-current.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if not r['ws_passed']:raise SystemExit(code or 90)
PY
exit "$code"

