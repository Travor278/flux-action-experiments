set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/py-spy" dump --pid 2162685 > "$R/status/download-v2-stack-1225.txt" 2>&1
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,os
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
proc=Path('/proc/2162685')
assert 'download_selected_v2.py' in (proc/'cmdline').read_bytes().decode().replace('\0',' ')
def snapshot():
 files=[]
 for p in (R/'official-source/.cache/huggingface/download').rglob('*'):
  if p.is_file() and ('file-094' in p.name or 'file-096' in p.name):
   st=p.stat();files.append({'path':str(p.relative_to(R)), 'bytes':st.st_size,'mtime':st.st_mtime})
 return {'time':time.time(),'io':(proc/'io').read_text(),'files':files}
a=[snapshot()];(R/'status/download-v2-target-samples.json').write_text(json.dumps(a));time.sleep(20)
a.append(snapshot());(R/'status/download-v2-target-samples.json').write_text(json.dumps(a));time.sleep(20)
a.append(snapshot());(R/'status/download-v2-target-samples.json').write_text(json.dumps(a))
PY