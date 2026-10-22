from pathlib import Path
import tarfile,hashlib,json,time
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
archive=B/'assets/pi05-ssh-data.tar';assert not archive.exists()
roots=[B/'assets/pi05-v3'/t for t in ['click_bell','handover_block']]+[B/'assets/index'/t for t in ['click_bell','handover_block']]+[B/'flux-action/src']
roots += [B/'scripts'/n for n in ['pi05_helpers.py','matched_window_sampler.py','train_pi05.py']]
roots += [B/'status'/n for n in ['pi05-data-verified.json','pi05-reference-environment.json']]
files=sorted({f for p in roots for f in ([p] if p.is_file() else p.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'})
manifest=[]
with tarfile.open(archive.with_suffix('.tar.tmp'),'w',dereference=True) as tar:
 for f in files:
  rel=f.relative_to(B).as_posix();tar.add(f,arcname=rel,recursive=False);manifest.append({'path':rel,'bytes':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()})
archive.with_suffix('.tar.tmp').rename(archive)
result={'files':manifest,'bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),'created_at':time.time()}
(B/'status/pi05-ssh-data-manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({'archive':str(archive),'bytes':result['bytes'],'sha256':result['sha256'],'files':len(files)}),flush=True)
