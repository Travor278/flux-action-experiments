set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,pickle,torch
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'));out=S/'runs/match_and_pick_from_conveyor-10k-prefix-v1';p=out/'step-1000'
assert (p/'COMPLETE').is_file();state=json.loads((p/'state.json').read_text());assert state['step']==1000 and state['world_size']==2
components={}
for name in ('model','optimizer','ema_0p10','ema_0p05'):
 with (p/name/'.metadata').open('rb') as f:m=pickle.load(f)
 components[name]=len(m.state_dict_metadata)
 for v in m.storage_data.values():
  q=(p/name/v.relative_path).resolve();assert q.is_relative_to((p/name).resolve()) and 0<=v.offset and v.length>0 and v.offset+v.length<=q.stat().st_size
assert components=={'model':451,'optimizer':1367,'ema_0p10':447,'ema_0p05':447}
for rank in (0,1):assert isinstance(torch.load(p/f'rng/rank-{rank:04d}.pt',weights_only=True),dict)
assert not (out/'step-719').exists()
record={'time':time.time(),'verified':True,'step':1000,'components':components,'rng_ranks':[0,1],'old_step719_removed_by_native_keep1':True,'manual_deleted_bytes':0,'keep_latest':str(p)}
(R/'status/conveyor-1000-checkpoint-audit.json').write_text(json.dumps(record,indent=2))
PY
