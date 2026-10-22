set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');run=Path(str(R).replace('/hdd/','/ssd/'))/'runs/perf-exact-20261014'
def tail(p,n=5000):
 if not p.exists():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
p=R/'status/perf-exact-current.json';r={'time':time.time(),'current':json.loads(p.read_text()) if p.exists() else None,'launch_tail':tail(run/'launch.log'),'variants':{}}
for mode in ['baseline','cached']:
 v={'log_tail':tail(run/f'{mode}.log'),'gpu_tail':tail(run/f'{mode}-gpu.csv',1000),'ranks':{}}
 for rank in (0,1):
  p=run/mode/f'comparison-rank{rank}.json'
  if p.exists():v['ranks'][rank]=json.loads(p.read_text())
  v[f'operators_rank{rank}']=tail(run/mode/f'rank{rank}-operators.txt',9000)
 r['variants'][mode]=v
(R/'status/perf-exact-monitor.json').write_text(json.dumps(r,indent=2))
PY
"$R/policy-venv/bin/python" "$R/scripts/monitor_formal_training.py"
