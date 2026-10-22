set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
intent={'time':time.time(),'reason':'User explicitly requests stopping custom prefix memory and restarting with official history conditioning. Preserve evidence; never resume custom runs automatically.','jobs':['job-REPLACE_WITH_YOUR_JOB_ID','job-REPLACE_WITH_YOUR_JOB_ID']}
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 p=S/'runs'/f'{task}-10k-prefix-v1';assert p.is_dir();(p/'STOP_AFTER_UPDATE').write_text(json.dumps(intent))
(R/'status/official-switch-stop-request.json').write_text(json.dumps(intent,indent=2))
print('OFFICIAL_SWITCH_STOP_REQUESTED')
PY
