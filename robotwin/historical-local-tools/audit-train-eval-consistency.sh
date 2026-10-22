set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time,numpy as np,h5py,zipfile,io,yaml,hashlib,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));T='click_bell';I=B/'assets/index'/T
train=json.loads((B/f'configs/{T}-10k-opt-v1.json').read_text());cfg=json.loads((S/f'runs/{T}-10k-opt-v1/export-10000-ema/config.json').read_text());meta=json.loads((B/f'assets/lerobot/{T}/meta/info.json').read_text());manifest=json.loads((I/'manifest.json').read_text());stats=json.loads((I/'statistics.json').read_text());protocol=json.loads((B/f'evaluation/{T}/flux-10k-test100/protocol.json').read_text());env=yaml.safe_load((B/'RoboTwin/env_cfg/task_config/demo_clean.yml').read_text())
keys=['camera_keys','camera_layout','canvas_hw','chunk_size','n_action_steps','fps','action_dim','action_modality','action_parameterization','gripper_flip_dims','augment','sampler','num_inference_steps','guidance_scale','guidance_scale_action','sampler_shift'];checks={k:train['policy'][k]==cfg[k] for k in keys};assert all(checks.values())
for field,block in [('action_normalization','action'),('state_normalization','state')]:assert cfg[field]=={k:stats[block][k] for k in ['q01','q99']}
assert meta['fps']==manifest['fps']==cfg['fps']==protocol['frequency']==15
assert meta['robot_type']==manifest['robot_type']=='aloha_agilex' and env['embodiment']==['aloha-agilex']
rows=np.load(I/'rows.f32.npy',mmap_mode='r');lengths=[];closeall=[];closestarts=[];active=[]
from flux_action.training.data import WindowDataset
D=WindowDataset(manifest,str(B/'assets/lerobot'/T),I/'rows.f32.npy',seed=42,decoder='pyav',frame_hw=train['frame_hw'])
for i,e in enumerate(manifest['episodes']):
 x=rows[e['from_index']:e['from_index']+e['n_frames'],:14];side=0 if np.ptp(x[:,:6],axis=0).max()>np.ptp(x[:,7:13],axis=0).max() else 1;col=6 if side==0 else 13;starts=D.valid_starts(i);lengths.append(e['n_frames']);closeall.extend((x[:,col]<.2).tolist());closestarts.extend((x[starts,col]<.2).tolist());active.append(side)
train_tasks=[json.loads(x)['task'] for x in (B/f'assets/lerobot/{T}/meta/tasks.jsonl').read_text().splitlines()];eval_pairs=json.loads((B/f'evaluation/{T}/flux-10k-test100/paired_seeds.json').read_text());train_set=set(train_tasks)
source_headers={};source_seen=set();total_seen=0;z=B/f'assets/robotwin-data/dataset/{T}/demo_clean.zip'
with zipfile.ZipFile(z) as archive:
 for k,n in enumerate(sorted(n for n in archive.namelist() if n.endswith('.hdf5'))):
  with h5py.File(io.BytesIO(archive.read(n))) as h:
   ins=json.loads(h['instructions'][()].decode());seen=ins['seen'] if isinstance(ins,dict) else ins;seen=[seen] if isinstance(seen,str) else seen;source_seen.update(seen);total_seen+=len(seen)
   if k==0:
    source_headers={'root_keys':list(h),'additional_info':{},'vision':{}}
    for name,item in h['additional_info'].items():
     if isinstance(item,h5py.Dataset):
      v=item[()];source_headers['additional_info'][name]=v.decode(errors='replace') if isinstance(v,bytes) else v.tolist() if isinstance(v,np.ndarray) else str(v)
    for cam,g in h['vision'].items():source_headers['vision'][cam]={name:{'shape':list(item.shape),'dtype':str(item.dtype)} for name,item in g.items() if isinstance(item,h5py.Dataset)}
overlap=[p for p in eval_pairs if p['instruction'] in train_set];source_overlap=[p for p in eval_pairs if p['instruction'] in source_seen]
report={'time':time.time(),'checks':checks,'normalization_matches_index':True,'robot_fps_protocol_match':True,'data_cameras':meta['features'],'source_headers':source_headers,'environment':env,'training_frames':meta['total_frames'],'train_episodes':len(lengths),'mean_episode_frames':float(np.mean(lengths)),'valid_window_starts':len(closestarts),'excluded_as_input_starts':meta['total_frames']-len(closestarts),'training_instruction_unique':len(train_set),'source_instruction_unique':len(source_seen),'source_instruction_total':total_seen,'eval_literal_train_overlap':len(overlap),'eval_literal_source_overlap':len(source_overlap),'eval_literal_overlap_successes':sum(p['success'] for p in overlap),'active_arm_demo_counts':{'left':active.count(0),'right':active.count(1)},'active_gripper_closed_all_frames_fraction':float(np.mean(closeall)),'active_gripper_closed_train_start_fraction':float(np.mean(closestarts)),'unproven':['No end-to-end expert-action replay against this exact simulator performed','Historical data-generation simulator assets/build not reconstructed bit-for-bit'],'success_criterion_source_sha':hashlib.sha256((B/'RoboTwin/envs/click_bell.py').read_bytes()).hexdigest()}
(B/'status/flux-train-eval-consistency-audit.json').write_text(json.dumps(report,indent=2))
PY
