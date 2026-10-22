import pathlib,json,os,time,hashlib,shutil,fcntl
from huggingface_hub import HfApi
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
lock=(B/'status/hf-final-upload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def retry(fn):
 for attempt in range(5):
  try:return fn()
  except Exception as e:
   from huggingface_hub.errors import RepositoryNotFoundError
   if isinstance(e,RepositoryNotFoundError):raise
   if attempt==4:raise
   print(json.dumps({'event':'HF_TRANSIENT_RETRY','attempt':attempt+1,'error_type':type(e).__name__}),flush=True);time.sleep(5)
proof=json.loads((B/'status/optimized-final-artifacts-verified.json').read_text());api=HfApi();assert retry(api.whoami)['name']=='Travor278'
status=B/'status/hf-final-uploads.json';r=json.loads(status.read_text()) if status.exists() else {'started':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 repo=f'YOUR_HF_ACCOUNT/flux-action-robotwin-{task.replace("_","-")}-10k-20261010';source=S/f'runs/{task}-10k-opt-v1/export-10000-ema';dest=S/f'hf-publish/{task}'
 if r['tasks'].get(task,{}).get('verified'):continue
 assert proof['tasks'][task]['verified'] and proof['tasks'][task]['exports']['10000']['verified']
 dest.mkdir(parents=True,exist_ok=True)
 for name in ['model.safetensors','config.json','manifest.json']:
  p=dest/name
  if not p.exists():os.link(source/name,p)
  assert os.path.samefile(source/name,p)
 card=f'''---
tags:
- robotics
- flux-action
- robotwin
- aloha-agilex
- flow-matching
base_model: black-forest-labs/flux-3-action-base
---
# FLUX Action — RoboTwin {task}, 10,000 updates

This private repository contains the experiment's final BF16 EMA (sigma_rel=0.10)
inference export. It is not the full optimizer/RNG training checkpoint.

## Training

- Task: `{task}`; Aloha-AgileX; 50 official RoboTwin `demo_clean` demonstrations.
- Dataset revision: `3dc3b798668feb99ac61cc9086d84cbcc3d79186`.
- Base: `black-forest-labs/flux-3-action-base@7f49687d2935d24aebe179c2d03bf7a356e00c0e`.
- Code: `black-forest-labs/flux-action@830f1ce178d7aea52f18fde31d8e8e36ce6bae21`.
- 10,000 optimizer updates; global batch 32 (320,000 sampled windows); seed 42.
- Full-parameter training; FP32 master weights/reduction and both FP32 CPU EMAs,
  BF16 computation; two H100s, microbatch 4/rank, accumulation 4.
- Native absolute joint/gripper actions (14 dimensions), RGB head/left/right wrist,
  chunk 32 and execute/replan 16, 15 Hz. No gripper flip, augmentation disabled.
- LR trunk 1e-5, action heads 5e-5; freeze 200, warmup 600/200, cooldown 9000–10000.
- Final logged training loss: {proof['tasks'][task]['final']['loss']:.8f}.

## Loading

Use the pinned FLUX Action code and `FluxActionPolicy.from_pretrained` with this
repository's downloaded directory. The saved config references the experiment's
local video VAE and text encoder. Download those assets from the pinned base model
and update `video_vae_id` and `text_encoder_id` in a local config copy to their paths.
The action weights and saved normalization come from this export, not the base.
See `manifest.json` for the export's source and SHA256 records.

## Evaluation status

Final checkpoints and all milestone exports were independently checked for SHA256
and SafeTensors readability. Closed-loop validation of this 10k checkpoint is in
progress; no success-rate claim is made here. Older initial 3k models used a different
training schedule and their scores must not be attributed to this model. Final
100-episode test scenarios remain reserved until checkpoint selection is complete.

Upstream code/model and RoboTwin data retain their respective licenses and terms.
'''
 (dest/'README.md').write_text(card)
 (dest/'training_recipe.json').write_text(json.dumps(json.loads((B/f'configs/{task}-10k-opt-v1.json').read_text()),indent=2))
 file_manifest={}
 for p in dest.iterdir():
  if p.is_file() and p.name in ['model.safetensors','config.json','manifest.json']:
   with p.open('rb') as f:file_manifest[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.file_digest(f,'sha256').hexdigest()}
 (B/f'status/hf-{task}-source-manifest.json').write_text(json.dumps(file_manifest,indent=2))
 known=task in r['tasks']
 r['tasks'].setdefault(task,{'repo_id':repo,'source':str(source),'staging':str(dest),'phase':'preparing','started':time.time()});status.write_text(json.dumps(r,indent=2))
 try:existing=retry(lambda:api.model_info(repo))
 except Exception as e:
  from huggingface_hub.errors import RepositoryNotFoundError
  if not isinstance(e,RepositoryNotFoundError):raise
  existing=None
 if existing is not None:assert known,'Unexpected preexisting repository; do not overwrite'
 retry(lambda:api.create_repo(repo_id=repo,repo_type='model',private=True,exist_ok=True))
 r['tasks'][task]={'repo_id':repo,'source':str(source),'staging':str(dest),'phase':'uploading','started':time.time()};status.write_text(json.dumps(r,indent=2))
 # The user explicitly requests checkpoint files only. No cards, recipes or scores.
 commit=retry(lambda:api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(dest),allow_patterns=['model.safetensors','config.json','manifest.json'],ignore_patterns=['.cache/**'],commit_message=f'Upload verified {task} 10k EMA inference checkpoint'))
 info=retry(lambda:api.model_info(repo,revision=commit.oid,files_metadata=True));assert info.private
 remote={x.rfilename:x for x in info.siblings}
 for name,spec in file_manifest.items():
  assert name in remote and remote[name].size==spec['bytes'],name
  if remote[name].lfs is not None:assert remote[name].lfs.sha256==spec['sha256'],name
 r['tasks'][task].update(verified=True,phase='verified',revision=commit.oid,url='https://huggingface.co/'+repo,files=file_manifest,finished=time.time());status.write_text(json.dumps(r,indent=2));print('HF_UPLOAD_VERIFIED '+repo,flush=True)
r.update(complete=True,completed=time.time());status.write_text(json.dumps(r,indent=2))
