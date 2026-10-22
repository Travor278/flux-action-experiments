set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$B/flux-action/.venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,subprocess
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B.parent/'flux_robodojo_20261013'
r={'time':time.time()}
for n in ['v3-inspection.log','sim-install.log','adapter-tests.log','official-history-tests.log']:
 p=R/'status'/n;r[n]=p.read_text(errors='replace')[-1500:] if p.exists() else None
r['processes']=subprocess.run(['ps','-ww','-eo','pid,etime,pcpu,args'],capture_output=True,text=True).stdout
r['processes']=[x for x in r['processes'].splitlines() if ('flux_robodojo_20261013' in x or 'inspect_robodojo_v3.py' in x or 'upload_tail_hf.py' in x) and 'ps -eo' not in x]
p=R/'status/official-v3-inspection.json'
if p.exists():
 j=json.loads(p.read_text());r['data']={k:j[k] for k in ['revision','tasks','episode_columns']};r['selected']={k:{'count':v['count'],'sample':v['rows'][:1]} for k,v in j['selected'].items()}
(R/'status/preparation-monitor.json').write_text(json.dumps(r,indent=2))
PY