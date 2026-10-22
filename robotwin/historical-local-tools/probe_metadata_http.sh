B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$B/flux-action/.venv/bin/python" - <<'PY' > "$R/status/metadata-http-probe.json" 2>&1
import httpx,time,json,urllib.parse
url='https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/35efbc7dedfdbeeb6e95fb749bd885d73d483e41/data/RoboDojo_lerobot_v30_video/meta/episodes/chunk-000/file-000.parquet'
r={'time':time.time()}
try:
 with httpx.Client(timeout=15,follow_redirects=True) as c:
  with c.stream('GET',url) as response:
   r.update(status=response.status_code,redirect_hosts=[urllib.parse.urlparse(str(x.url)).hostname for x in response.history],host=urllib.parse.urlparse(str(response.url)).hostname)
   r['first_chunk_bytes']=len(next(response.iter_bytes(65536)))
except Exception as e:r['error']=type(e).__name__
print(json.dumps(r))
PY