set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,shutil
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1'
r={'time':time.time(),'output':str(out),'exists':out.exists(),'free_ssd_bytes':shutil.disk_usage(out.parent if out.parent.exists() else out.parents[1]).free}
for name in ['train.log','resume2.log','resume3.log','export.log','gpu.csv','gpu-recovery.csv']:
 p=out/name;r[name]=p.read_text(errors='replace')[-4500:] if p.exists() else None
p=out/'metrics.jsonl';rows=[]
if p.exists():
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
r['recent_metrics']=rows[-10:];r['checkpoint_markers']=[str(p.relative_to(out)) for p in out.glob('step-*/COMPLETE')]
r['exact_resume']={str(rank):json.loads((out/f'exact-resume-rank{rank}.json').read_text()) if (out/f'exact-resume-rank{rank}.json').exists() else None for rank in (0,1)}
p=R/'status/full-preflight-complete.json';r['complete']=json.loads(p.read_text()) if p.exists() else None
(R/'status/full-preflight-monitor.json').write_text(json.dumps(r,indent=2))
PY