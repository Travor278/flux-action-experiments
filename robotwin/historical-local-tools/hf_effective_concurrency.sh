set -euo pipefail
python3 - <<'PY'
import pathlib,re,json,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));logs=sorted((S/'hf-transfer-cache/logs').glob('*.log'),key=lambda p:p.stat().st_mtime)[-2:];out={'time':time.time(),'logs':[]}
for p in logs:
 s=p.read_text(errors='replace');matches=re.findall(r'Current concurrency[^\d]*(\d+)',s)
 out['logs'].append({'path':str(p),'bytes':p.stat().st_size,'observed_concurrency':list(dict.fromkeys(map(int,matches))),'mtime':p.stat().st_mtime})
(B/'status/hf-effective-concurrency.json').write_text(json.dumps(out,indent=2))
PY
