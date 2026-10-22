set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" "$R/scripts/monitor_official_history.py" > "$R/status/monitor-20261021.log" 2>&1
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/cover-charger-inventory-20261021.json"
from pathlib import Path
import json,pyarrow.parquet as pq,shutil,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
root=R/'official-source/data/RoboDojo_lerobot_v30_video'
paths=list((root/'meta/episodes').rglob('*.parquet')); rows=[]
for p in paths:
 for row in pq.read_table(p).to_pylist():
  if any('charger' in str(t).lower() for t in row.get('tasks',[])):rows.append({k:v for k,v in row.items() if not k.startswith('stats/')})
result={'time':time.time(),'meta_files':[str(p.relative_to(R)) for p in paths],'charger_episodes':rows,'free_ssd':shutil.disk_usage(S).free,'free_hdd':shutil.disk_usage(R).free,'old_runs':{p.name:{'complete':(p/'TRAIN_COMPLETE.json').exists(),'checkpoints':[x.parent.name for x in p.glob('step-*/COMPLETE')]} for p in (S/'runs').glob('*official-history-v1')}}
print(json.dumps(result,indent=2))
PY
