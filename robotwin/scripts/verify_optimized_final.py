import pathlib,json,time,hashlib,math,statistics
from torch.distributed.checkpoint import FileSystemReader
from safetensors import safe_open
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 R=S/f'runs/{task}-10k-opt-v1';c=R/'step-10000';assert (R/'TRAIN_COMPLETE.json').exists() and (c/'COMPLETE').exists()
 st=json.loads((c/'state.json').read_text());assert st['step']==10000 and st['world_size']==2
 assert len(list((c/'rng').glob('rank-*.pt')))==2
 groups={}
 for name in ['model','optimizer','ema_0p10','ema_0p05']:
  p=c/name;m=FileSystemReader(str(p)).read_metadata();assert m.storage_data;files={}
  for storage in m.storage_data.values():
   f=p/storage.relative_path;assert f.resolve().is_relative_to(p.resolve()) and f.is_file()
   size=f.stat().st_size;assert size>=storage.offset+storage.length;files[f.name]=size
  groups[name]={'entries':len(m.state_dict_metadata),'files':len(files),'bytes':sum(files.values())}
 exports={}
 for step in [1000,3000,5000,10000]:
  e=R/f'export-{step}-ema';m=json.loads((e/'manifest.json').read_text());assert m['weight_profile']=='ema_0p10'
  receipt=json.loads((R/f'export-{step}-verified.json').read_text());assert receipt['step']==step and receipt['verified']
  for name,h in m['sha256'].items():
   with (e/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
  cfg=json.loads((e/'config.json').read_text());assert cfg['action_dim']==14 and cfg['n_action_steps']==16 and cfg['chunk_size']==32 and cfg['torch_dtype']=='bfloat16'
  with safe_open(str(e/'model.safetensors'),framework='pt',device='cpu') as f:assert len(f.keys())==451
  exports[str(step)]={'path':str(e),'sha256':m['sha256'],'tensors':451,'verified':True}
 rows=[json.loads(x) for x in (R/'metrics.jsonl').read_text().splitlines()];updates=[r for r in rows if 'step_time' in r]
 assert updates[-1]['step']==10000 and all(math.isfinite(r[k]) for r in updates for k in ['loss','grad_norm'])
 out['tasks'][task]={'verified':True,'final':updates[-1],'checkpoint_groups':groups,'exports':exports,'resume_steps':[r['step'] for r in rows if r.get('event')=='resume'],'sampled_windows':320000}
out['time']=time.time();(B/'status/optimized-final-artifacts-verified.json').write_text(json.dumps(out,indent=2));print('BOTH_OPTIMIZED_FINALS_AND_8_EXPORTS_SHA_VERIFIED',flush=True)
