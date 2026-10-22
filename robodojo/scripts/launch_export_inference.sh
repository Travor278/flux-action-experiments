#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
RUN="$R/status/export-inference-jobs/$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)"
mkdir -p "$RUN"
export FLUX_EXPORT_PROBE_DIR="$RUN" PYTHONPATH="$R/scripts:$R/flux-action/src" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec > >(tee -a "$RUN/launch.log") 2>&1
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.used,utilization.gpu --format=csv -l 5 > "$RUN/gpu.csv" &
mon=$!
trap 'kill "$mon" 2>/dev/null || true' EXIT
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,os,socket
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');run=Path(os.environ['FLUX_EXPORT_PROBE_DIR'])
r={'time':time.time(),'hostname':socket.gethostname(),'run':str(run),'verified':False,'phase':'starting'}
(run/'preflight.json').write_text(json.dumps(r));(R/'status/export-inference-current.json').write_text(json.dumps(r))
PY
set +e
timeout --signal=TERM --kill-after=20s 1200s "$R/policy-venv/bin/python" "$R/scripts/probe_export_inference.py" > "$RUN/probe.log" 2>&1
code=$?
set -e
export EXPORT_PROBE_EXIT_CODE="$code"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,os
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');run=Path(os.environ['FLUX_EXPORT_PROBE_DIR']);r=json.loads((run/'preflight.json').read_text());code=int(os.environ['EXPORT_PROBE_EXIT_CODE'])
r.update(finished=time.time(),exit_code=code,verified=code==0 and (run/'export-inference-verified.json').exists())
r['phase']='passed' if r['verified'] else 'failed';(run/'result.json').write_text(json.dumps(r,indent=2));(R/'status/export-inference-current.json').write_text(json.dumps(r,indent=2))
if not r['verified']:raise SystemExit(code or 90)
PY
