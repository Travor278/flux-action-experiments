set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,hashlib
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
g=json.loads((R/'status/perf-exact-ready.json').read_text());assert g['ready']
for n,h in g['files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
out=Path(g['outputs']['match_and_pick_from_conveyor'])
assert not (out/'TRAIN_COMPLETE.json').exists()
intent={'time':time.time(),'purpose':'User-authorized speed optimization with frozen scientific parameters; reuse conveyor two H100 for exact comparison then restore original production checkpoint','old_job':'job-REPLACE_WITH_YOUR_JOB_ID','resume_launcher':str(R/'scripts/launch_profile_then_resume.sh'),'no_cache_adoption':True}
(R/'status/perf-exact-stop-intent.json').write_text(json.dumps(intent,indent=2))
if not (out/'STOP_AFTER_UPDATE').exists():(out/'STOP_AFTER_UPDATE').write_text(json.dumps(intent))
print('CONVEYOR_SAFE_STOP_REQUESTED',flush=True)
PY
