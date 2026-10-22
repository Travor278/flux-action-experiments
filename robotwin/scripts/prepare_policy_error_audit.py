"""Read-only coverage audit and fixed teacher-forced policy probes."""
import hashlib, io, json, pathlib, sys, time, zipfile
import h5py
import numpy as np
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
sys.path.insert(0,str(B))
from XPolicyLab.utils.process_data import decode_image_bit
from flux_action.training.data import WindowDataset
R=B/'evaluation/policy-error-audit-v1';R.mkdir(exist_ok=True)
joint_keys=['left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states']
selected=[r['episode'] for r in json.loads((B/'evaluation/expert-replay-click/plan.json').read_text())['episodes']]
coverage={};probes=[]
for task in ['click_bell','handover_block']:
 index=B/'assets/index'/task;manifest=json.loads((index/'manifest.json').read_text())
 D=WindowDataset(manifest,str(B/'assets/lerobot'/task),index/'rows.f32.npy',seed=42,decoder='pyav',frame_hw=[240,320])
 data=[]
 with zipfile.ZipFile(B/f'assets/robotwin-data/dataset/{task}/demo_clean.zip') as z:
  files=sorted(n for n in z.namelist() if n.endswith('.hdf5'))
  scene=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/scene_info.json'))))
  for ep,name in enumerate(files):
   with h5py.File(io.BytesIO(z.read(name))) as h:
    state=np.concatenate([h[f'state/{k}'][()] for k in joint_keys],axis=1).astype(np.float32)
    action=np.concatenate([h[f'action/{k}'][()] for k in joint_keys],axis=1).astype(np.float32)
    n=len(state);starts=np.array(D.valid_starts(ep));mask=np.zeros(n,dtype=bool);mask[starts]=True
    closed=(state[:,[6,13]]<.2)
    moving=scene[f'episode_{ep}'].get('info',{}).get('{a}') if task=='click_bell' else None
    side=0 if moving=='left' else 1 if moving=='right' else None
    relevant=closed[:,side] if side is not None else closed.any(1)
    row={'episode':ep,'frames':n,'valid_starts':len(starts),'left_closed_all':int(closed[:,0].sum()),'right_closed_all':int(closed[:,1].sum()),
         'relevant_closed_all':int(relevant.sum()),'relevant_closed_starts':int(relevant[starts].sum()),
         'closed_start_probability_within_episode':float(relevant[starts].mean()),'side':moving}
    data.append(row)
    if ep not in (selected if task=='click_bell' else list(range(10))):continue
    src=h['instructions'][()];captions=json.loads(src.decode());captions=captions['seen'] if isinstance(captions,dict) else captions
    caption=manifest['episodes'][ep]['caption'];assert isinstance(caption,str)
    alternate=next(t for t in captions if t!=caption)
    closed_indices=np.flatnonzero(relevant);first_closed=int(closed_indices[0]) if len(closed_indices) else n-16
    arm='left' if side==0 else 'right';ee=h[f'state/{arm}_ee_poses'][()];bottom=int(closed_indices[np.argmin(ee[closed_indices,2])]) if len(closed_indices) else int(np.argmin(ee[:,2]))
    if task=='click_bell':
     anchors={'early':1,'middle_valid':max(1,int(starts[-1])//2),'last_valid':int(starts[-1]),'first_closed':first_closed,'lowest_ee':bottom,'tail_16':n-16}
     joints=list(range(6)) if side==0 else list(range(7,13));grips=[6 if side==0 else 13]
    else:
     lc=np.flatnonzero(closed[:,0]);rc=np.flatnonzero(closed[:,1]);left=int(lc[0]) if len(lc) else n//3;right=int(rc[0]) if len(rc) else 2*n//3
     release=np.flatnonzero((np.arange(n)>right)&(state[:,6]>.8)&closed[:,1]);release=int(release[0]) if len(release) else n-16
     anchors={'early':1,'left_close':left,'right_close':right,'handover_release':release,'last_valid':int(starts[-1]),'tail_16':n-16}
     arm='both';joints=list(range(6))+list(range(7,13));grips=[6,13]
    for phase,t in anchors.items():
     t=max(1,min(t,n-2));case=f'{task}-ep{ep:02d}-{phase}';payload={'state':state[t],'target':action[t:min(t+32,n)]}
     for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:payload[cam]=decode_image_bit(h[f'vision/{cam}/colors'][t])
     path=R/f'{case}.npz';np.savez_compressed(path,**payload)
     probes.append({'id':case,'task':task,'episode':ep,'phase':phase,'frame':t,'valid_training_start':bool(mask[t]),'target_horizon':len(payload['target']),
                    'side':arm,'gripper_indices':grips,'active_joint_indices':joints,
                    'train_caption':caption,'alternate_caption':alternate,'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 total=sum(x['frames'] for x in data);valid=sum(x['valid_starts'] for x in data)
 coverage[task]={'episodes':len(data),'frames':total,'valid_observation_starts':valid,'eligible_fraction':valid/total,
  'relevant_closed_all_fraction':sum(x['relevant_closed_all'] for x in data)/total,
  'relevant_closed_all_starts_fraction':sum(x['relevant_closed_starts'] for x in data)/valid,
  'relevant_closed_actual_sampler_probability':float(np.mean([x['closed_start_probability_within_episode'] for x in data])),
  'weighting':'sampler chooses episodes uniformly, then valid starts uniformly within the episode','per_episode':data}
plan={'time':time.time(),'version':1,'checkpoint_step':10000,'scope':'teacher-forced diagnostics on training demonstration observations; not held-out success rate',
 'selected_source_episodes':selected,'inference_steps':[4,16],'caption_variants':['train','alternate'],'inference_seed':0,
 'expected_predictions':len(probes)*4,'coverage':coverage,'probes':probes,'mutates_training_data':False}
(R/'plan.json').write_text(json.dumps(plan,indent=2));(B/'status/policy-error-audit-ready.json').write_text(json.dumps({'time':time.time(),'complete':True,'probes':len(probes),'predictions':len(probes)*4,'plan_sha256':hashlib.sha256((R/'plan.json').read_bytes()).hexdigest(),'coverage':{t:{k:v for k,v in x.items() if k!='per_episode'} for t,x in coverage.items()}},indent=2))
print('POLICY_ERROR_AUDIT_INPUTS_READY',flush=True)
