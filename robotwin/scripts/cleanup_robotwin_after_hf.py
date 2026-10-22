from pathlib import Path
import os,json,time,hashlib,fcntl,shutil,subprocess
from huggingface_hub import HfApi
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=Path(str(B).replace('/hdd/','/ssd/'))
lock=(B/'status/robotwin-close.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
u=json.loads((B/'status/hf-tail-uploads.json').read_text());assert u['complete']
receipt=json.loads((B/'status/robotwin-local-records-receipt.json').read_text());manifest=json.loads((B/'status/robotwin-records-manifest.json').read_text());assert receipt['sha256']==manifest['sha256'] and receipt['six_records_verified']
pid=int((B/'status/hf-tail-upload.pid').read_text());assert not Path(f'/proc/{pid}').exists(),'Uploader still alive'
api=HfApi()
for task,r in u['tasks'].items():
 assert r['verified'];info=api.model_info(r['repo_id'],revision=r['revision'],files_metadata=True);assert info.private
 files={f.rfilename:f for f in info.siblings};assert set(files)=={'.gitattributes',*r['files']}
 for n,v in r['files'].items():
  assert files[n].size==v['bytes']
  if files[n].lfs:assert files[n].lfs.sha256==v['sha256']
targets=[S/'runs']+[B/'assets'/n for n in ['robotwin-sim-assets','robotwin-data','pi05-v3','lerobot','index','index-tail-v1']]
targets += [p for p in [S/'hf-tail-transfer-cache',S/'hf-tail-metadata-cache'] if p.exists()]
for p in targets:
 assert not p.is_symlink() and p.resolve().is_relative_to(B if p.is_relative_to(B) else S) and p not in [B,S]
references=[]
for proc in Path('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  links=[]
  for q in [proc/'cwd',*list((proc/'fd').iterdir())]:
   try:links.append(os.readlink(q))
   except OSError:pass
  for p in targets:
   if str(p) in cmd or any(x==str(p) or x.startswith(str(p)+'/') for x in links):references.append({'pid':proc.name,'target':str(p)})
 except (FileNotFoundError,ProcessLookupError):pass
assert not references,references
result={'time':time.time(),'hf_verified':u,'local_records':receipt,'deleted':[],'free_before':shutil.disk_usage(S).free}
for p in targets:
 if not p.exists():continue
 before=int(subprocess.check_output(['du','-s','-B1',str(p)],text=True).split()[0])
 shutil.rmtree(p)
 assert not p.exists();result['deleted'].append({'path':str(p),'allocated_bytes':before})
 (B/'status/robotwin-close-cleanup.json').write_text(json.dumps(result,indent=2))
result.update(complete=True,finished=time.time(),free_after=shutil.disk_usage(S).free,total_allocated_bytes=sum(x['allocated_bytes'] for x in result['deleted']))
(B/'status/robotwin-close-cleanup.json').write_text(json.dumps(result,indent=2));print('CLEANUP_COMPLETE',flush=True)
