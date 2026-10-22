set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec > "$R/status/step2-audit.log" 2>&1
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,pickle,hashlib,shutil,torch
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1';p=out/'step-2';s=json.loads((p/'state.json').read_text());assert s['step']==2 and s['world_size']==2 and (p/'COMPLETE').exists()
evidence=R/'status/full-preflight-step2-evidence';evidence.mkdir(exist_ok=True)
for n in ['state.json','config.json','COMPLETE']:shutil.copy2(p/n,evidence/n)
checks={}
for name in ['model','optimizer','ema_0p10','ema_0p05']:
 with (p/name/'.metadata').open('rb') as f:m=pickle.load(f)
 assert m.state_dict_metadata
 for v in m.storage_data.values():
  file=(p/name/v.relative_path).resolve();assert file.is_relative_to((p/name).resolve()) and file.is_file()
  assert 0<=v.offset and v.length>0 and v.offset+v.length<=file.stat().st_size
 checks[name]={'keys':len(m.state_dict_metadata),'entries':len(m.storage_data),'metadata_sha256':hashlib.sha256((p/name/'.metadata').read_bytes()).hexdigest()}
 shutil.copy2(p/name/'.metadata',evidence/(name+'-metadata.pkl'))
for rank in (0,1):
 rng=p/f'rng/rank-{rank:04d}.pt';r=torch.load(rng,weights_only=True);assert 'torch' in r;shutil.copy2(rng,evidence/rng.name)
backup=R/'status/full-preflight-attempt1-worker.py';assert hashlib.sha256(backup.read_bytes()).hexdigest()=='dc90f317879932d798b8a19ff3b6a54e35b28648139df5003fb1fd4ece0bb3b6'
(R/'status/full-preflight-step2-independent-audit.json').write_text(json.dumps({'time':time.time(),'verified':True,'step':2,'world_size':2,'components':checks,'state_sha256':hashlib.sha256((p/'state.json').read_bytes()).hexdigest(),'data_position':s['data']['position'],'scope':'DCP metadata/complete marker/shard bounds/two RNG readable; no original in-memory hashes due diagnostic helper ImportError'},indent=2))
print('STEP2_CHECKPOINT_STRUCTURAL_AUDIT_PASS')
PY