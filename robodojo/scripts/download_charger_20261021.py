from pathlib import Path
import json,time,hashlib,os,fcntl
from huggingface_hub import HfApi,hf_hub_download
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
revision='35efbc7dedfdbeeb6e95fb749bd885d73d483e41';prefix='data/RoboDojo_lerobot_v30_video'
lock=(R/'status/charger-download-20261021.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
eps=json.loads((R/'status/cover-charger-inventory-20261021.json').read_text())['charger_episodes']
assert sorted(e['episode_index'] for e in eps)==list(range(2200,2300))
paths=set()
for e in eps:
 paths.add(f"{prefix}/data/chunk-{e['data/chunk_index']:03d}/file-{e['data/file_index']:03d}.parquet")
 for camera in ['cam_high','cam_left_wrist','cam_right_wrist']:
  k='videos/observation.images.'+camera;paths.add(f"{prefix}/{k}/chunk-{e[k+'/chunk_index']:03d}/file-{e[k+'/file_index']:03d}.mp4")
api=HfApi();info=api.get_paths_info('RoboDojo-Benchmark/RoboDojo',sorted(paths),repo_type='dataset',revision=revision)
specs=[{'path':f.path,'bytes':f.size,'sha256':f.lfs.sha256 if f.lfs else None} for f in info]
assert len(specs)==len(paths) and all(s['sha256'] for s in specs)
plan={'time':time.time(),'revision':revision,'episodes':eps,'files':specs,'bytes':sum(s['bytes'] for s in specs)}
(R/'status/charger-download-plan-20261021.json').write_text(json.dumps(plan,indent=2))
temp=R/'charger-source-20261021.incoming';final=R/'charger-source-20261021';temp.mkdir(exist_ok=True)
done={}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for spec in specs:
 rel=spec['path'];dest=(final if final.exists() else temp)/rel
 dest.parent.mkdir(parents=True,exist_ok=True)
 existing=R/'official-source'/rel
 if not dest.exists() and existing.exists() and existing.stat().st_size==spec['bytes'] and sha(existing)==spec['sha256']:os.link(existing,dest)
 if not dest.exists():
  assert not final.exists()
  dest=Path(hf_hub_download('RoboDojo-Benchmark/RoboDojo',rel,repo_type='dataset',revision=revision,local_dir=temp))
 assert dest.stat().st_size==spec['bytes'] and sha(dest)==spec['sha256']
 done[rel]={'bytes':spec['bytes'],'sha256':spec['sha256']}
 (R/'status/charger-download-progress-20261021.json').write_text(json.dumps({'time':time.time(),'verified_files':len(done),'total_files':len(specs),'verified_bytes':sum(s['bytes'] for s in done.values()),'total_bytes':plan['bytes']}))
 print('VERIFIED',rel,flush=True)
if not final.exists():temp.rename(final)
(R/'status/charger-download-verified-20261021.json').write_text(json.dumps({'time':time.time(),'complete':True,'source':str(final),'revision':revision,'episodes':list(range(2200,2300)),'files':done},indent=2))
print('CHARGER_DOWNLOAD_VERIFIED',flush=True)
