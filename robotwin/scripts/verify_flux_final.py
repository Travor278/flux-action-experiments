from pathlib import Path
import json,hashlib,time,subprocess
from torch.distributed.checkpoint import FileSystemReader
from safetensors import safe_open
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
out={'time':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 root=S/'runs'/task;c=root/'step-3000';e=root/'export-3000-ema'
 assert (root/'TRAIN_COMPLETE').exists() and (c/'COMPLETE').exists(),task
 state=json.loads((c/'state.json').read_text());assert state['step']==3000 and state['world_size']==8
 assert len(list((c/'rng').glob('rank-*.pt')))==8
 groups={}
 for group in ['model','optimizer','ema_0p10','ema_0p05']:
  p=c/group;meta=FileSystemReader(str(p)).read_metadata();assert meta.storage_data
  files={}
  for storage in meta.storage_data.values():
   f=p/storage.relative_path;assert f.resolve().is_relative_to(p.resolve()) and f.is_file()
   size=f.stat().st_size;assert size>=storage.offset+storage.length;files[str(f.name)]=size
  groups[group]={'tensor_entries':len(meta.state_dict_metadata),'shards':len(files),'bytes':sum(files.values())}
 manifest=json.loads((e/'manifest.json').read_text());assert manifest['weight_profile']=='ema_0p10'
 for name,h in manifest['sha256'].items():
  with (e/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
 with safe_open(e/'model.safetensors',framework='pt',device='cpu') as sf:keys=sf.keys();assert len(keys)>400
 cfg=json.loads((e/'config.json').read_text());assert cfg['action_dim']==14 and cfg['n_action_steps']==16 and cfg['chunk_size']==32 and cfg['torch_dtype']=='bfloat16'
 out['tasks'][task]={'steps':3000,'world_size':8,'checkpoint':str(c),'export':str(e),'export_sha256':manifest['sha256'],'export_tensors':len(keys),'checkpoint_groups':groups,'verified':True}
 print(task,'TRAIN_AND_EXPORT_VERIFIED',flush=True)
out['gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
(B/'status/flux-final-artifacts-verified.json').write_text(json.dumps(out,indent=2))
