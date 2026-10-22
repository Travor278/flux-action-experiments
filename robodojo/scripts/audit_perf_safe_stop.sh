set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,pickle,torch
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'));out=S/'runs/match_and_pick_from_conveyor-10k-prefix-v1'
r={'time':time.time(),'stopped':False}
p=out/'STOPPED_SAFELY.json'
if p.exists():
 stop=json.loads(p.read_text());cp=max(out.glob('step-*/COMPLETE'),key=lambda x:int(x.parent.name.split('-')[1])).parent;stop['step']=int(cp.name.split('-')[1]);assert (cp/'COMPLETE').exists()
 components={}
 for n in ('model','optimizer','ema_0p10','ema_0p05'):
  with (cp/n/'.metadata').open('rb') as f:m=pickle.load(f)
  components[n]=len(m.state_dict_metadata)
  for v in m.storage_data.values():
   target=(cp/n/v.relative_path).resolve();assert target.is_relative_to((cp/n).resolve()) and v.offset+v.length<=target.stat().st_size
 assert components=={'model':451,'optimizer':1367,'ema_0p10':447,'ema_0p05':447}
 for rank in (0,1):assert isinstance(torch.load(cp/f'rng/rank-{rank:04d}.pt',weights_only=True),dict)
 r.update(stopped=True,verified=True,step=stop['step'],checkpoint=str(cp),components=components)
(R/'status/perf-safe-stop-verified.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
PY
"$R/policy-venv/bin/python" "$R/scripts/monitor_formal_training.py"
