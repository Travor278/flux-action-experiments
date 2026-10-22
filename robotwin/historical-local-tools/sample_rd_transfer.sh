set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$B/flux-action/.venv/bin/python" - <<'PY' > "$R/status/transfer-sampling.log" 2>&1
import pathlib,json,time,os
R=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=pathlib.Path(str(R).replace('/hdd/','/ssd/'))
def snap():
 r={'time':time.time(),'processes':{},'partial':[]}
 for pid in [17663,17679]:
  p=pathlib.Path(f'/proc/{pid}')
  if p.exists():r['processes'][str(pid)]={'io':(p/'io').read_text(),'wchan':(p/'wchan').read_text()}
 for root in [R/'uv-cache',R/'official-source/.cache',S/'hf-download-cache']:
  if not root.exists():continue
  fs=[]
  for p in root.rglob('*'):
   if p.is_file() and not p.is_symlink():
    st=p.stat()
    if st.st_size>1000000 or p.suffix=='.incomplete':fs.append({'path':str(p),'size':st.st_size,'mtime':st.st_mtime})
  r['partial']+=sorted(fs,key=lambda x:x['mtime'],reverse=True)[:12]
 return r
r={'first':snap()};time.sleep(20);r['second']=snap();(R/'status/transfer-samples.json').write_text(json.dumps(r,indent=2))
PY