R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/sim-venv/bin/python" - <<'PY' > "$R/status/conveyor-cpu-usd.log" 2>&1
from pathlib import Path
import json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
try:
 from pxr import Usd
 stage=Usd.Stage.Open(str(R/'official-source/Assets/Object/RoboDojo/Dynamic/conveyor/00000/object.usdz'))
 rows=[]
 for p in stage.Traverse():
  typ=p.GetAttribute('node:type').Get() if p.HasAttribute('node:type') else ''
  if typ or 'Graph' in p.GetTypeName():
   rows.append({'path':str(p.GetPath()),'type':p.GetTypeName(),'node_type':str(typ),'attrs':{a.GetName():{'value':str(a.Get()),'connections':[str(c) for c in a.GetConnections()]} for a in p.GetAttributes()},'relationships':{r.GetName():[str(t) for t in r.GetTargets()] for r in p.GetRelationships()}})
 result={'time':time.time(),'cpu_usd_available':True,'graph_nodes':rows,'scope':'Read-only source USD, noSimulationApp orGPUcontext started'}
except Exception as e:result={'time':time.time(),'cpu_usd_available':False,'error':repr(e)}
(R/'status/conveyor-cpu-usd.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
PY
