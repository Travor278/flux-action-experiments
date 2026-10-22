R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/conveyor-source-inspection.log" 2>&1
from pathlib import Path
import json,zipfile,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
p=R/'official-source/Assets/Object/RoboDojo/Dynamic/conveyor/00000/object.usdz'
with zipfile.ZipFile(p) as z:
 records=[]
 for f in z.infolist():
  if f.filename.endswith(('.usda','.usd','.usdc')) and f.file_size<1000000:
   data=z.read(f);text=data.decode('utf-8',errors='replace')
   records.append({'name':f.filename,'bytes':f.file_size,'header':text[:100],'matching_lines':[l for l in text.splitlines() if any(s in l.lower() for s in ['velocity','conveyor','inputs:','onplayback','delay'])][:100]})
 r={'time':time.time(),'entries':[{'name':f.filename,'bytes':f.file_size} for f in z.infolist()][:25],'small_usd':records}
 (R/'status/conveyor-source-inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
PY
