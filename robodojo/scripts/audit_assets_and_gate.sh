R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/selected-download-independent-audit.log" 2>&1
from pathlib import Path
import json,time,hashlib,os
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
plan=json.loads((R/'status/selected-download-plan.json').read_text());proof=json.loads((R/'status/selected-download-verified.json').read_text())
assert proof['complete'] and proof['revision']==plan['revision'] and proof['selected_episode_ids']==plan['selected_episode_ids']
assert len(proof['files'])==len(plan['files'])==15400
ledger={v['path']:v for v in map(json.loads,(R/'status/selected-download-receipts.jsonl').read_text().splitlines())}
assert set(ledger)==set(proof['files'])=={s['path'] for s in plan['files']}
for s in plan['files']:
 p=R/'official-source'/s['path'];v=proof['files'][s['path']];stat=p.stat()
 assert stat.st_size==v['bytes']==s['bytes'] and stat.st_mtime_ns==v['mtime_ns']
 assert ledger[s['path']]['sha256']==v['sha256']
 if s['lfs_sha256']:assert v['sha256']==s['lfs_sha256']
assert (R/'RoboDojo/Assets').resolve()==(R/'official-source/Assets').resolve()
cmd=Path('/proc/2162685/cmdline');assert not cmd.exists() or 'download_selected_v2.py' not in cmd.read_text()
r={'time':time.time(),'verified':True,'files':15400,'bytes':sum(v['bytes'] for v in proof['files'].values()),'revision':proof['revision'],'receipt_sha256':hashlib.sha256((R/'status/selected-download-verified.json').read_bytes()).hexdigest(),'checks':'Plan/receipt/ledger exact membership; source LFS SHA matches existing per-file hashes; all current sizes and mtimes unchanged; writer exited; Assets symlink correct. Full48GB not rehashed redundantly.'}
(R/'status/selected-download-independent-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
PY
"$R/policy-venv/bin/python" "$R/scripts/prepare_scene_control_gate.py" > "$R/status/scene-control-gate.log" 2>&1
