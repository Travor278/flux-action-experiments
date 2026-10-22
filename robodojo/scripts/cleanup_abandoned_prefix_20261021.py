from pathlib import Path
import json,time,fcntl,shutil,hashlib,pickle
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
audit=json.loads((R/'status/official-switch-stopped-verified.json').read_text());assert audit['verified']
receipt=R/'status/abandoned-prefix-cleanup-20261021.json'
assert not receipt.exists(),'Already cleaned; inspect receipt instead'
locks=[];targets=[]
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 lock=(R/f'status/formal-{task}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
 run=(S/'runs'/f'{task}-10k-prefix-v1').resolve();assert run.is_relative_to((S/'runs').resolve())
 assert (run/'STOP_AFTER_UPDATE').exists(),'Only explicitly abandoned prefix runs allowed'
 expected=Path(audit['records'][task]['checkpoint']).resolve()
 for cp in run.glob('step-*'):
  assert cp.resolve()==expected and cp.is_dir() and not cp.is_symlink()
  assert (cp/'COMPLETE').exists()
  evidence=R/'status/abandoned-prefix-evidence-20261021'/task/cp.name;evidence.mkdir(parents=True,exist_ok=False)
  files=[];size=0
  for p in cp.rglob('*'):
   assert not p.is_symlink(),p
   if p.is_file():
    n=p.stat().st_size;size+=n;files.append({'path':str(p.relative_to(cp)),'bytes':n})
    if n<10_000_000:
     dest=evidence/p.relative_to(cp);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
     assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest()
  for name in ['model','optimizer','ema_0p10','ema_0p05']:
   with (cp/name/'.metadata').open('rb') as f:m=pickle.load(f)
   for x in m.storage_data.values():assert x.offset+x.length<=(cp/name/x.relative_path).stat().st_size
  (evidence/'file-sizes.json').write_text(json.dumps(files,indent=2))
  targets.append({'checkpoint':str(cp),'bytes':size,'evidence':str(evidence)})
intent={'time':time.time(),'reason':'User abandoned custom prefix memory and requested fresh official Cover+Charger plus timely cleanup of unused checkpoints/optimizers','targets':targets,'free_before':shutil.disk_usage(S).free}
(R/'status/abandoned-prefix-cleanup-intent-20261021.json').write_text(json.dumps(intent,indent=2))
for item in targets:
 p=Path(item['checkpoint']);assert p.resolve().is_relative_to((S/'runs').resolve());shutil.rmtree(p);assert not p.exists()
intent.update(deleted=True,bytes_deleted=sum(x['bytes'] for x in targets),free_after=shutil.disk_usage(S).free)
receipt.write_text(json.dumps(intent,indent=2));print(json.dumps(intent),flush=True)
