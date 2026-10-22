"""Remote-only selective official v3 and simulator assets; no model downloads."""
from pathlib import Path
import json,hashlib,time,fcntl,concurrent.futures,os
from huggingface_hub import HfApi,hf_hub_download
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
lock=(R/'status/selected-download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
meta=json.loads((R/'status/official-v3-inspection.json').read_text());rev=meta['revision'];prefix=meta['prefix'];repo='RoboDojo-Benchmark/RoboDojo';api=HfApi()
expected=set();selected_ids={}
for task,group in meta['selected'].items():
 assert group['count']==100,(task,group['count'])
 selected_ids[task]=[int(e['episode_index']) for e in group['rows']]
 for e in group['rows']:
  expected.add(prefix+'/'+meta['info']['data_path'].format(chunk_index=int(e['data/chunk_index']),file_index=int(e['data/file_index'])))
  for key,spec in meta['info']['features'].items():
   if spec['dtype']!='video':continue
   expected.add(prefix+'/'+meta['info']['video_path'].format(video_key=key,chunk_index=int(e[f'videos/{key}/chunk_index']),file_index=int(e[f'videos/{key}/file_index'])))
inventory=json.loads((R/'status/preparation.json').read_text());assert inventory['dataset_revision']==rev
asset_files=[f for f in inventory['remote_files'] if f['path'].startswith('Assets/')]
data_files=[]
for f in api.list_repo_tree(repo,path_in_repo=prefix,repo_type='dataset',revision=rev,recursive=True):
 if not hasattr(f,'size'):continue
 if f.path in expected:data_files.append({'path':f.path,'bytes':f.size,'lfs_sha256':f.lfs.sha256 if f.lfs else None})
assert {f['path'] for f in data_files}==expected
files=data_files+asset_files
plan={'time':time.time(),'revision':rev,'selected_episode_ids':selected_ids,'files':files,'bytes':sum(f['bytes'] for f in files)}
(R/'status/selected-download-plan.json').write_text(json.dumps(plan,indent=2))
done={};status=R/'status/selected-download-progress.json'
def one(spec):
 path=Path(hf_hub_download(repo,spec['path'],repo_type='dataset',revision=rev,local_dir=R/'official-source'))
 assert path.stat().st_size==spec['bytes']
 with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 if spec['lfs_sha256']:assert digest==spec['lfs_sha256'],spec['path']
 return spec['path'],{'bytes':spec['bytes'],'sha256':digest}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for future in concurrent.futures.as_completed([pool.submit(one,f) for f in files]):
  k,v=future.result();done[k]=v
  status.write_text(json.dumps({'time':time.time(),'total_files':len(files),'verified_files':len(done),'verified_bytes':sum(x['bytes'] for x in done.values()),'total_bytes':plan['bytes']}))
(R/'status/selected-download-verified.json').write_text(json.dumps({'time':time.time(),'complete':True,'revision':rev,'selected_episode_ids':selected_ids,'files':done},indent=2))
assets=R/'RoboDojo/Assets'
if not assets.exists():assets.symlink_to(R/'official-source/Assets',target_is_directory=True)
print('SELECTED_DATA_AND_ASSETS_VERIFIED',flush=True)
