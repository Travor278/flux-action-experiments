#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
RUN="$R/status/real-vae-jobs/$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)"
mkdir -p "$RUN"
export FLUX_VAE_PROBE_DIR="$RUN" PYTHONPATH="$R/scripts:$R/flux-action/src" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
exec > >(tee -a "$RUN/launch.log") 2>&1
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.used,utilization.gpu --format=csv -l 5 > "$RUN/gpu.csv" 2>&1 &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import os,json,socket,time,subprocess
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(os.environ['FLUX_VAE_PROBE_DIR'])
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(gpu)==1 and '4090' in gpu[0]
r={'time':time.time(),'hostname':socket.gethostname(),'run':str(out),'gpu':gpu,'phase':'starting','verified':False}
(out/'preflight.json').write_text(json.dumps(r,indent=2));(R/'status/real-vae-current.json').write_text(json.dumps(r,indent=2))
PY
set +e
timeout --signal=TERM --kill-after=20s 900s "$R/policy-venv/bin/python" "$R/scripts/probe_real_vae.py" > "$RUN/probe.log" 2>&1
code=$?
set -e
export PROBE_EXIT_CODE="$code"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import os,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(os.environ['FLUX_VAE_PROBE_DIR']);r=json.loads((out/'preflight.json').read_text());code=int(os.environ['PROBE_EXIT_CODE']);marker=out/'real-vae-verified.json'
r.update(finished=time.time(),exit_code=code,verified=code==0 and marker.exists(),phase='passed' if code==0 and marker.exists() else 'failed')
if r['verified']:assert json.loads(marker.read_text())['verified']
(out/'result.json').write_text(json.dumps(r,indent=2));(R/'status/real-vae-current.json').write_text(json.dumps(r,indent=2))
if not r['verified']:raise SystemExit(code or 90)
PY
