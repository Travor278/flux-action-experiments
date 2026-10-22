R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import subprocess,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
processes=subprocess.check_output(['ps','-ww','-eo','pid,ppid,etime,stat,args'],text=True)
(R/'status/step2-audit-process-check.json').write_text(json.dumps({'time':time.time(),'processes':[l for l in processes.splitlines() if 'flux_robodojo' in l or 'python -' in l],'files':[p.name for p in (R/'status/full-preflight-step2-evidence').glob('*')]},indent=2))
PY