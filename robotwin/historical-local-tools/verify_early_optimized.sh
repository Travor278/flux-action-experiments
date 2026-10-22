set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time
from torch.distributed.checkpoint import FileSystemReader
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 root=S/f'runs/{task}-10k-opt-v1';c=root/'step-100'
 assert (c/'COMPLETE').exists() and (root/'EARLY_CHECKPOINT_VERIFIED.json').exists()
 state=json.loads((c/'state.json').read_text());assert state['step']==100 and state['world_size']==2
 assert len(list((c/'rng').glob('rank-*.pt')))==2
 groups={}
 for name in ['model','optimizer','ema_0p10','ema_0p05']:
  p=c/name;m=FileSystemReader(str(p)).read_metadata();assert m.storage_data;files={}
  for storage in m.storage_data.values():
   f=p/storage.relative_path;assert f.resolve().is_relative_to(p.resolve()) and f.is_file()
   size=f.stat().st_size;assert size>=storage.offset+storage.length;files[f.name]=size
  groups[name]={'entries':len(m.state_dict_metadata),'files':len(files),'bytes':sum(files.values())}
 out['tasks'][task]={'verified':True,'step':100,'world_size':2,'groups':groups}
(B/'status/optimized-early-checkpoints-audit.json').write_text(json.dumps(out,indent=2))
PY
