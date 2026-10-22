set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'));out=S/'runs/match_and_pick_from_conveyor-10k-prefix-v1'
rows=[]
for p in out.glob('step-*'):
 if not p.is_dir():continue
 files=[f for f in p.rglob('*') if f.is_file()]
 rows.append({'path':str(p),'complete':(p/'COMPLETE').exists(),'bytes':sum(f.stat().st_size for f in files),'files':len(files)})
(R/'status/perf-stop-residue-audit.json').write_text(json.dumps({'time':time.time(),'rows':rows,'deleted_bytes':0,'protected_step':719},indent=2))
PY
