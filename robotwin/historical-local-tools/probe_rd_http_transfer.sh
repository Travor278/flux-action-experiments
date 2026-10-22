B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
from pathlib import Path
import json,httpx,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');plan=json.loads((R/'status/selected-download-plan.json').read_text());f=next(x for x in plan['files'] if x['path'].startswith('data/') and x['path'].endswith('.mp4'))
url='https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/'+plan['revision']+'/'+f['path'];r={'time':time.time(),'path':f['path'],'probe_only':True}
t=time.monotonic();count=0
try:
 with httpx.Client(timeout=httpx.Timeout(20,connect=10),follow_redirects=True) as client:
  with client.stream('GET',url,headers={'Range':'bytes=0-8388607'}) as response:
   r['status']=response.status_code;response.raise_for_status()
   for chunk in response.iter_bytes(1048576):
    count+=len(chunk)
    if count>=8388608:break
 r.update(bytes=count,seconds=time.monotonic()-t)
except Exception as e:r.update(error=type(e).__name__,bytes=count,seconds=time.monotonic()-t)
(R/'status/http-download-probe.json').write_text(json.dumps(r,indent=2))
PY