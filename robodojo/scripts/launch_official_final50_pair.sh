#!/usr/bin/env bash
set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec 9> "$R/status/official-final50-pair.lock"
flock -n 9 || exit 3
for task in cover_blocks match_and_pick_from_conveyor; do
 bash "$R/scripts/launch_official_final50.sh" "$task" "${1:-}"
done
"$R/policy-venv/bin/python" - <<'PY'
import json,time
from pathlib import Path
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
results={}
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 r=json.loads((R/'evaluation'/task/'flux-10k-official-history-v1-final50/COMPLETE.json').read_text());assert r['verified'] and r['total']==50
 results[task]={'successes':r['successes'],'total':50}
(R/'status/official-final50-pair-complete.json').write_text(json.dumps({'time':time.time(),'verified':True,'tasks':results},indent=2))
PY
