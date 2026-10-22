#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
RUN="$R/status/renderer-jobs/$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launch.log") 2>&1
export FLUX_RENDER_STATUS_DIR="$RUN"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,subprocess,os,time,socket,hashlib
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(os.environ['FLUX_RENDER_STATUS_DIR'])
q=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(q)==1,q
name,driver,memory=[v.strip() for v in q[0].split(',')];assert '4090' in name
record={'time':time.time(),'hostname':socket.gethostname(),'gpu':name,'driver':driver,'memory_mib':int(memory),'run':str(out),'script_sha256':hashlib.sha256((R/'scripts/render_smoke_lab.py').read_bytes()).hexdigest(),'phase':'driver_verified','render_passed':False}
(out/'preflight.json').write_text(json.dumps(record,indent=2));(R/'status/renderer-job-current.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
major=int(driver.split('.')[0])
if major<570 or major>=595:
 record['phase']='driver_rejected';record['reason']='Driver outside candidate range for IsaacSim5.1; no repeated RTX crash on known595 branch'
 (out/'result.json').write_text(json.dumps(record,indent=2));(R/'status/renderer-job-current.json').write_text(json.dumps(record,indent=2));raise SystemExit(78)
PY
unset CUDA_VISIBLE_DEVICES VK_DRIVER_FILES
if [ -s /etc/vulkan/icd.d/nvidia_icd.json ]; then export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; elif [ -s /usr/share/vulkan/icd.d/nvidia_icd.json ]; then export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; fi
export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$R/xdg-cache" XDG_CONFIG_HOME="$R/xdg-config" XDG_DATA_HOME="$R/xdg-data"
set +e
timeout --signal=TERM --kill-after=15s 480s "$R/sim-venv/bin/python" "$R/scripts/render_smoke_lab.py" > "$RUN/render.log" 2>&1
code=$?
set -e
export RENDER_EXIT_CODE="$code"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import os,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(os.environ['FLUX_RENDER_STATUS_DIR']);r=json.loads((out/'preflight.json').read_text());code=int(os.environ['RENDER_EXIT_CODE']);p=out/'renderer-smoke-verified.json'
r.update(finished=time.time(),exit_code=code,phase='render_passed' if code==0 and p.exists() else 'render_failed',render_passed=code==0 and p.exists())
if r['render_passed']:r['render']=json.loads(p.read_text());assert r['render']['verified']
(out/'result.json').write_text(json.dumps(r,indent=2));(R/'status/renderer-job-current.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if not r['render_passed']:raise SystemExit(code or 90)
PY
exit "$code"
