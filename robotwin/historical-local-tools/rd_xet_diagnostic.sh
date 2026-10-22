B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
from pathlib import Path
import json,re,time,collections
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'));logs=sorted((S/'hf-download-cache').rglob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:3];out={'time':time.time(),'logs':[]}
for p in logs:
 text=p.read_text(errors='replace')[-300000:]
 out['logs'].append({'name':p.name,'mtime':p.stat().st_mtime,'size':p.stat().st_size,'timeouts':len(re.findall('timed out|Timeout|os error 110',text)),'concurrency':re.findall(r'Current concurrency[^\n]{0,70}',text)[-5:],'retry_mentions':text.lower().count('retry'),'errors':re.findall(r'"level"\s*:\s*"(ERROR|WARN)"',text)[-20:]})
(R/'status/xet-download-diagnostic.json').write_text(json.dumps(out,indent=2))
PY