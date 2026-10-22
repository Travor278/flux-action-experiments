R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/scene-first-failure-evidence.log" 2>&1
from pathlib import Path
import hashlib,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
c=json.loads((R/'status/scene-control-current.json').read_text());run=Path(c['run']);assert 'job-09f45428-' in str(run)
for name in ['probe_scene_control.py','launch_scene_control.sh']:
 src=R/'scripts'/name;(run/('original-'+name)).write_bytes(src.read_bytes())
logs={}
for f in run.glob('*.log'):
 logs[str(f)]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
kit=R/'sim-venv/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1'
for n in ['kit_20261013_170157.log','kit_20261013_170321.log']:
 f=kit/n
 if f.exists():
  text=f.read_text(errors='replace');lines=text.splitlines()
  logs[n]={'bytes':f.stat().st_size,'matches':[x for x in lines if any(k in x for k in ['Traceback','Error','Exception','ModuleNotFound','ImportError'])][-25:],'tail':lines[-12:]}
r={'time':time.time(),'job':'job-REPLACE_WITH_YOUR_JOB_ID','current':c,'logs':logs,'scope':'First failure preserved before diagnostic exception logging repair. Rootcause not yet proven.'}
(R/'status/scene-first-failure-evidence.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
PY
