R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/post-download-cleanup.log" 2>&1
from pathlib import Path
import json,time,fcntl,os
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
assert json.loads((R/'status/selected-download-independent-audit.json').read_text())['verified']
lock=(R/'status/selected-download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
root=(R/'official-source/.cache/huggingface/download').resolve()
assert root.is_relative_to(R.resolve())
items=[]
for p in root.rglob('*.incomplete'):
 assert p.resolve().is_relative_to(root) and not p.is_symlink()
 items.append({'path':str(p),'bytes':p.stat().st_size});p.unlink()
r={'time':time.time(),'verified_complete_download':True,'deleted':items,'bytes':sum(x['bytes'] for x in items),'scope':'Only owned obsolete HF .incomplete transfer fragments under exclusive completed-download lock; source/cache metadata retained'}
(R/'status/post-download-cleanup.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
PY
