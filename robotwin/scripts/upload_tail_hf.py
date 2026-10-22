import pathlib, os, json, time, hashlib, fcntl
from huggingface_hub import HfApi, hf_hub_download, set_client_factory
import httpx
from huggingface_hub.errors import RepositoryNotFoundError
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
lock=(B/'status/hf-tail-upload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
set_client_factory(lambda:httpx.Client(timeout=httpx.Timeout(120,connect=15),follow_redirects=True))
def retry(fn):
 for attempt in range(5):
  try:return fn()
  except (httpx.TransportError,httpx.HTTPStatusError) as e:
   if attempt==4:raise
   print(json.dumps({'retry':attempt+1,'error':type(e).__name__}),flush=True);time.sleep(5)
api=HfApi();assert retry(api.whoami)['name']=='Travor278'
proof=json.loads((B/'status/tail-final-artifacts-verified.json').read_text());assert proof['complete']
status=B/'status/hf-tail-uploads.json'
r=json.loads(status.read_text()) if status.exists() else {'started':time.time(),'tasks':{}}
def save():
 p=status.with_suffix('.tmp');p.write_text(json.dumps(r,indent=2));p.replace(status)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for task in ['click_bell','handover_block']:
 if r['tasks'].get(task,{}).get('verified'):continue
 source=S/f'runs/{task}-10k-tail-v1/export-10000-ema'
 repo=f'YOUR_HF_ACCOUNT/flux-action-robotwin-{task.replace("_","-")}-10k-tail-v1-20261013'
 files={n:{'bytes':(source/n).stat().st_size,'sha256':sha(source/n)} for n in ['model.safetensors','config.json','manifest.json']}
 for n,v in proof['tasks'][task]['files'].items():assert files[n]['sha256']==v['sha256'] and files[n]['bytes']==v['size']
 known=task in r['tasks']
 try:
  api.model_info(repo);assert known,'Unexpected existing repository'
 except RepositoryNotFoundError:pass
 r['tasks'][task]={'repo_id':repo,'source':str(source),'files':files,'phase':'uploading','started':time.time()};save()
 api.create_repo(repo,private=True,exist_ok=True)
 commit=retry(lambda:api.upload_folder(repo_id=repo,folder_path=str(source),allow_patterns=list(files),commit_message='Upload verified repaired RoboTwin 10k EMA checkpoint'))
 info=api.model_info(repo,revision=commit.oid,files_metadata=True);assert info.private
 remote={x.rfilename:x for x in info.siblings};assert set(remote)==set(files)|{'.gitattributes'}
 for n,v in files.items():
  assert remote[n].size==v['bytes']
  if remote[n].lfs:assert remote[n].lfs.sha256==v['sha256']
  else:
   p=pathlib.Path(hf_hub_download(repo,n,revision=commit.oid,cache_dir=str(S/'hf-tail-metadata-cache')));assert sha(p)==v['sha256']
 r['tasks'][task].update(verified=True,phase='verified',revision=commit.oid,finished=time.time(),url='https://huggingface.co/'+repo);save()
 print('VERIFIED '+repo,flush=True)
r.update(complete=True,finished=time.time());save()
