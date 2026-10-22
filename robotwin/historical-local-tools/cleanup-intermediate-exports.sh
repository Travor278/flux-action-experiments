set -euo pipefail
python3 - <<'PY'
import pathlib,json,hashlib,shutil,time,os
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));base=(S/'runs').resolve();hf=json.loads((B/'status/hf-checkpoint-only-verified.json').read_text());assert hf['complete'];assert (B/'status/flux-milestone-stop-verified.txt').exists()
finals={}
for task in ['click_bell','handover_block']:
 f=S/f'runs/{task}-10k-opt-v1/export-10000-ema/model.safetensors';s=f.stat();assert s.st_size==13894319944;finals[task]=(s.st_ino,s.st_size)
paths=[S/f'runs/{t}-10k-opt-v1/export-{n}-ema' for t in ['click_bell','handover_block'] for n in [1000,3000,5000]]
for d in paths:assert not d.is_symlink() and d.resolve().is_relative_to(base) and d.resolve().parent.name in ['click_bell-10k-opt-v1','handover_block-10k-opt-v1']
refs=[]
for pp in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(pp/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');maps=(pp/'maps').read_text(errors='replace')
  for d in paths:
   if str(d) in cmd or str(d) in maps:refs.append({'pid':int(pp.name),'path':str(d),'kind':'cmdline/maps'})
  for fd in (pp/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if any(target.startswith(str(d)+'/') for d in paths):refs.append({'pid':int(pp.name),'fd':str(fd),'target':target})
   except (FileNotFoundError,PermissionError,OSError):pass
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not refs,refs
record={'time':time.time(),'reason':'User requested cleanup of intermediate checkpoints; only final10k evaluation remains','deleted':[],'logical_bytes':0,'allocated_bytes_estimate':0,'active_references':refs,'finals_preserved':list(finals)}
for d in paths:
 fs=list(d.iterdir());assert {p.name for p in fs}=={'model.safetensors','config.json','manifest.json'} and all(p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 for p in fs)
 manifest=json.loads((d/'manifest.json').read_text());assert hashlib.sha256((d/'config.json').read_bytes()).hexdigest()==manifest['sha256']['config.json']
 E=B/'status/intermediate-export-cleanup-evidence'/d.parent.name/d.name;E.mkdir(parents=True,exist_ok=True)
 for name in ['config.json','manifest.json']:shutil.copy2(d/name,E/name)
 item={'path':str(d),'bytes':sum(p.stat().st_size for p in fs),'allocated_bytes':sum(p.stat().st_blocks*512 for p in fs),'manifest':manifest};record['deleted'].append(item);record['logical_bytes']+=item['bytes'];record['allocated_bytes_estimate']+=item['allocated_bytes']
(B/'status/flux-intermediate-export-cleanup-before.json').write_text(json.dumps(record,indent=2))
for d in paths:
 for name in ['model.safetensors','config.json','manifest.json']:(d/name).unlink()
 d.rmdir()
for task,identity in finals.items():
 s=(S/f'runs/{task}-10k-opt-v1/export-10000-ema/model.safetensors').stat();assert (s.st_ino,s.st_size)==identity
record['complete']=True;record['completed_at']=time.time();record['disk_free_bytes_after']=shutil.disk_usage(S).free
(B/'status/flux-intermediate-export-cleanup.json').write_text(json.dumps(record,indent=2))
PY
