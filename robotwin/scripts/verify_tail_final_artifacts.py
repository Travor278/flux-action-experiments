import pathlib,json,time,hashlib,pickle,math,shutil,os
from safetensors import safe_open
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
r={'time':time.time(),'complete':False,'tasks':{}}
for task in ['click_bell','handover_block']:
 R=S/f'runs/{task}-10k-tail-v1';E=R/'export-10000-ema';C=R/'step-10000'
 done=json.loads((R/'TRAIN_COMPLETE.json').read_text());assert done['step']==10000 and done['complete']
 assert (C/'COMPLETE').is_file();state=json.loads((C/'state.json').read_text());assert state['step']==10000 and state['world_size']==2
 dcp={}
 for part in ['model','optimizer','ema_0p10','ema_0p05']:
  with (C/part/'.metadata').open('rb') as f:m=pickle.load(f)
  assert m.state_dict_metadata;dcp[part]=len(m.state_dict_metadata)
  for st in m.storage_data.values():assert (C/part/st.relative_path).stat().st_size>=st.offset+st.length
 assert len(list((C/'rng').glob('rank-*.pt')))==2
 manifest=json.loads((E/'manifest.json').read_text());files={}
 for n,h in manifest['sha256'].items():
  with (E/n).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
  assert actual==h,(task,n);files[n]={'size':(E/n).stat().st_size,'sha256':actual}
 with safe_open(str(E/'model.safetensors'),framework='pt',device='cpu') as f:
  keys=list(f.keys());assert len(keys)==451
  assert all(all(d>0 for d in f.get_slice(k).get_shape()) for k in keys)
 cfg=json.loads((E/'config.json').read_text());assert cfg['action_dim']==14 and cfg['chunk_size']==32 and cfg['n_action_steps']==16
 metrics=[]
 for line in (R/'metrics.jsonl').read_text().splitlines():
  x=json.loads(line)
  if isinstance(x,dict) and 'loss' in x:metrics.append(x)
 assert metrics[-1]['step']==10000
 assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) and x['windows']==32 for x in metrics)
 assert all(y['step']>x['step'] for x,y in zip(metrics,metrics[1:]))
 r['tasks'][task]={'complete':True,'step':10000,'last_update':metrics[-1],'logged_updates':len(metrics),'dcp':dcp,'export':str(E),'files':files,'tensor_count':len(keys),'complete_time':done['time'],'config':{k:cfg.get(k) for k in ['action_dim','chunk_size','n_action_steps','num_inference_steps','fps']}}
r['complete']=True;r['finished']=time.time();r['free_ssd']=shutil.disk_usage(S).free
r['cleanup_candidates']=[str(p) for root in [B/'assets',S] for p in root.iterdir() if p.is_file() and p.suffix in ['.partial','.part','.tmp','.tar']]
(B/'status/tail-final-artifacts-verified.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
