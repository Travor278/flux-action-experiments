from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,time,os,shutil
R=Path('/home/USER/codex_remote_ops/flux_robotwin_pi05_20261008').resolve()
I=Path('/home/USER/rpent-data/codex_remote_ops/flux_robotwin_pi05_20261008').resolve()
assert str(R)=='/home/USER/codex_remote_ops/flux_robotwin_pi05_20261008'
assert str(I)=='/home/coder/data/rpent-travor/codex_remote_ops/flux_robotwin_pi05_20261008'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(root,manifest):
 for f in manifest['files']:
  p=root/f['path'];assert p.resolve().is_relative_to(root) and not p.is_symlink()
  assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256'],str(p)
def unpack(archive,root,allowed):
 with tarfile.open(archive,'r') as tar:
  for m in tar.getmembers():
   rel=PurePosixPath(m.name);assert not rel.is_absolute() and '..' not in rel.parts and m.isfile(),m.name
   assert m.name in allowed,m.name
   dest=root/m.name;assert dest.resolve().is_relative_to(root),str(dest)
   dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():continue
   with tar.extractfile(m) as src,dest.open('wb') as out:shutil.copyfileobj(src,out,8*1024*1024)
manifest=json.loads((I/'pi05-ssh-data-manifest.json').read_text(encoding='utf-8-sig'));archive=I/'pi05-ssh-data.tar'
assert archive.stat().st_size==1763051520
assert digest(archive)==manifest['sha256']=='9de1b340ea4cbecb9307b158711f7394adafc2c5c327e4ca96ce6bd5ad9c7ee9'
unpack(archive,R,{x['path'] for x in manifest['files']});verify(R,manifest)
print('DATA_80_FILES_SHA_PASS',flush=True)
toktar=I/'pi05-tokenizer.tar';tokroot=R/'assets/paligemma-tokenizer';tokroot.mkdir(parents=True,exist_ok=True)
with tarfile.open(toktar) as t:tokmanifest=json.load(t.extractfile('verified_manifest.json'))
assert tokmanifest['revision']=='35e4f46485b4d07967e7e9935bc3786aad50687c'
unpack(toktar,tokroot,{x['path'] for x in tokmanifest['files']}|{'verified_manifest.json'});verify(tokroot,tokmanifest)
print('TOKENIZER_7_FILES_SHA_PASS',flush=True)
model=R/'assets/pi05-base';modelmanifest=json.loads((model/'verified_manifest.json').read_text());assert modelmanifest['revision']=='b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba';verify(model,modelmanifest)
print('MODEL_DESTINATION_SHA_PASS',flush=True)
from safetensors import safe_open
with safe_open(model/'model.safetensors',framework='pt',device='cpu') as sf:assert len(sf.keys())>100
from transformers import AutoTokenizer
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
tok=AutoTokenizer.from_pretrained(str(tokroot),local_files_only=True);assert len(tok.encode('press the bell'))>1
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
reads=[]
for task,total in [('click_bell',3855),('handover_block',14057)]:
 ds=LeRobotDataset(task,root=R/'assets/pi05-v3'/task,video_backend='pyav');assert ds.meta.total_episodes==50 and len(ds)==total
 for idx in [0,total-1]:
  row=ds[idx];assert tuple(row['action'].shape)==(14,) and tuple(row['observation.state'].shape)==(14,)
  assert torch.isfinite(row['action']).all() and torch.isfinite(row['observation.state']).all()
  for key in ['observation.images.cam_head','observation.images.cam_left_wrist','observation.images.cam_right_wrist']:
   assert tuple(row[key].shape)==(3,240,320) and torch.isfinite(row[key]).all()
 reads.append({'task':task,'episodes':50,'frames':total,'boundary_rows':[0,total-1],'three_cameras_shape':[3,240,320],'state_action_dim':14})
result={'time':time.time(),'verified':True,'data_files':len(manifest['files']),'archive_sha256':manifest['sha256'],'model_revision':modelmanifest['revision'],'tokenizer_revision':tokmanifest['revision'],'tokenizer_files':len(tokmanifest['files']),'datasets':reads,'training_started':False}
(R/'status/ssh-transfer-verified.json').write_text(json.dumps(result,indent=2));(R/'status/model-tokenizer-ready.json').write_text(json.dumps({'time':time.time(),'verified':True}))
removed=[]
for p in [archive,toktar]:
 assert p.parent==I and not p.is_symlink();removed.append({'path':str(p),'bytes':p.stat().st_size});p.unlink()
partial=R/'assets/.paligemma-tokenizer.download'
if partial.exists():
 pidfile=R/'status/download.pid';pid=int(pidfile.read_text()) if pidfile.exists() else 0
 active=Path(f'/proc/{pid}/cmdline')
 if not active.exists() or b'download_pi05_ssh' not in active.read_bytes():
  assert partial.resolve().is_relative_to(R) and not partial.is_symlink();shutil.rmtree(partial);removed.append({'path':str(partial)})
(R/'status/ssh-transfer-cleanup.json').write_text(json.dumps({'time':time.time(),'removed':removed},indent=2))
print(json.dumps(result),flush=True)
