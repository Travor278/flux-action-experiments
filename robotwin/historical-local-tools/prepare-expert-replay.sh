set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,zipfile,io,h5py,numpy as np,json,hashlib,time
from XPolicyLab.utils.process_data import decode_image_bit
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/expert-replay-click';R.mkdir(exist_ok=True);keys=['left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states'];records=[]
with zipfile.ZipFile(B/'assets/robotwin-data/dataset/click_bell/demo_clean.zip') as z:
 names=sorted(n for n in z.namelist() if n.endswith('.hdf5'));seeds=[int(s) for s in z.read(next(n for n in z.namelist() if n.endswith('/seed.txt'))).split()];scenes=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/scene_info.json'))));assert len(names)==len(seeds)==50
 selected=[]
 for arm in ['left','right']:
  selected.extend([i for i in range(50) if scenes[f'episode_{i}']['info']['{a}']==arm][:5])
 for i in sorted(selected):
  raw=z.read(names[i]);a={}
  with h5py.File(io.BytesIO(raw)) as h:
   for g in ['state','action']:a[g]=np.concatenate([h[f'{g}/{k}'][()] for k in keys],axis=1).astype('float32')
   for arm in ['left','right']:a[f'{arm}_ee_poses']=h[f'state/{arm}_ee_poses'][()]
   for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:
    a[f'{cam}_rgb0']=decode_image_bit(h[f'vision/{cam}/colors'][0]);a[f'{cam}_intrinsic0']=h[f'vision/{cam}/intrinsic_matrix'][0];a[f'{cam}_extrinsic0']=h[f'vision/{cam}/extrinsics_matrix'][0]
   assert int(h['additional_info/frequency'][()])==15
  assert a['state'].shape==a['action'].shape and a['state'].shape[1]==14
  np.savez_compressed(R/f'episode-{i}.npz',**a)
  records.append({'episode':i,'seed':seeds[i],'source_member':names[i],'source_hdf5_sha256':hashlib.sha256(raw).hexdigest(),'frames':len(a['action']),'source_scene':scenes[f'episode_{i}'],'npz':str(R/f'episode-{i}.npz')})
plan={'time':time.time(),'task':'click_bell','episodes':records,'selection':'first five left-arm and first five right-arm source demonstrations, fixed before results','source_archive_revision':'3dc3b798668feb99ac61cc9086d84cbcc3d79186','frequency':15,'joint_order':keys,'models_used':False}
(R/'plan.json').write_text(json.dumps(plan,indent=2));(B/'status/expert-replay-plan-summary.json').write_text(json.dumps({'episodes':[{'episode':r['episode'],'seed':r['seed'],'arm':r['source_scene']['info']['{a}'],'frames':r['frames']} for r in records],'balanced_source_seeds':True,'no_model':True}))
PY
