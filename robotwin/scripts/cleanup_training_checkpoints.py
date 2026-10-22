from pathlib import Path
import json,shutil,time,re
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008').resolve();S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008').resolve()
proof=json.loads((B/'status/flux-final-artifacts-verified.json').read_text());receipt={'time':time.time(),'removed':[],'preserved_exports':[]}
for task in ['click_bell','handover_block']:
 assert (B/f'status/flux-debug-{task}-ready.txt').read_text().strip()=='PLAIN_AND_ENCODED_DEBUG_PASS'
 root=S/'runs'/task;assert (root/'TRAIN_COMPLETE').exists() and proof['tasks'][task]['verified']
 export=root/'export-3000-ema';assert export.is_dir() and (export/'model.safetensors').is_file()
 dest=B/'status/final-checkpoint-metadata'/task;dest.mkdir(parents=True,exist_ok=True)
 for p in root.glob('step-*'):
  assert re.fullmatch(r'step-\d+',p.name) and p.resolve().is_relative_to(root) and not p.is_symlink()
  assert (p/'COMPLETE').exists(),'Refuse active/incomplete checkpoint'
  for name in ['state.json','config.json','COMPLETE']:shutil.copy2(p/name,dest/(p.name+'-'+name))
  files=[f for f in p.rglob('*') if f.is_file() and not f.is_symlink()]
  entry={'task':task,'path':str(p),'files':len(files),'bytes':sum(f.stat().st_size for f in files)}
  shutil.rmtree(p);receipt['removed'].append(entry)
 receipt['preserved_exports'].append(str(export))
receipt['bytes_removed']=sum(x['bytes'] for x in receipt['removed'])
(B/'status/training-checkpoint-cleanup.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
