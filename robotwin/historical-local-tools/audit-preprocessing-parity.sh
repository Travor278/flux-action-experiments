set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
OMP_NUM_THREADS=2 "$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,torch,numpy as np,time,types,zipfile,h5py,io,yaml,sys
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));sys.path.insert(0,str(B));torch.set_num_threads(2)
from flux_action.training.data import WindowDataset,collate_windows
from flux_action.config import PolicyConfig
from flux_action.policy import FluxActionPolicy
from flux_action.processing import packing
from XPolicyLab.policy.FLUX_Action.model import observation_arrays,action_dicts,STATE_KEYS,CAMERAS
I=B/'assets/index/click_bell';m=json.loads((I/'manifest.json').read_text());cfg=PolicyConfig(**json.loads((S/'runs/click_bell-10k-opt-v1/export-10000-ema/config.json').read_text()));d=WindowDataset(m,str(B/'assets/lerobot/click_bell'),I/'rows.f32.npy',seed=42,decoder='pyav',frame_hw=[240,320]);item=d.window_at(0,1);batch=collate_windows([item]);s=item['state'].numpy();dims=[6,1,6,1];edges=np.cumsum([0]+dims);obs={'instruction':item['task'],'state':{k:s[edges[i]:edges[i+1]] for i,k in enumerate(STATE_KEYS)},'vision':{camera:{'color':item[key][0].numpy().transpose(1,2,0)} for key,camera in CAMERAS.items()}}
a=observation_arrays([obs],dims);runtime={k:v if k=='task' else torch.from_numpy(v) for k,v in a.items()};dummy=types.SimpleNamespace(config=cfg,_resize_camera=FluxActionPolicy._resize_camera)
training_camera=FluxActionPolicy._cameras(dummy,batch)[:,:,:1];runtime_camera=FluxActionPolicy._cameras(dummy,runtime);assert torch.equal(training_camera,runtime_camera) and np.array_equal(a['state'][0],s)
x=packing.materialize_video(training_camera[0],None,'cpu',layout=cfg.camera_layout,canvas_hw=cfg.canvas_hw);y=packing.materialize_video(runtime_camera[0],None,'cpu',layout=cfg.camera_layout,canvas_hw=cfg.canvas_hw);assert torch.equal(x,y)
normalized=FluxActionPolicy._action_targets(dummy,item['action'],None);decoded=FluxActionPolicy._targets_to_actions(dummy,normalized,item['state']);err=float(torch.max(torch.abs(decoded-item['action'])));assert err<1e-5
parts=action_dicts(item['action'].numpy()[None],dims,16)[0];joined=np.stack([np.concatenate([r[k] for k in STATE_KEYS]) for r in parts]);assert np.array_equal(joined,item['action'].numpy()[:16])
sourceK={}
with zipfile.ZipFile(B/'assets/robotwin-data/dataset/click_bell/demo_clean.zip') as z:
 name=sorted(n for n in z.namelist() if n.endswith('.hdf5'))[0]
 with h5py.File(io.BytesIO(z.read(name))) as h:
  for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:sourceK[cam]=h[f'vision/{cam}/intrinsic_matrix'][0].tolist()
c=yaml.safe_load((B/'RoboTwin/env_cfg/task_config/_camera_config.yml').read_text())['D435'];f=c['h']/2/np.tan(np.deg2rad(c['fovy'])/2);expected=np.array([[f,0,c['w']/2],[0,f,c['h']/2],[0,0,1]]);errors={k:float(np.max(np.abs(np.array(K)-expected))) for k,K in sourceK.items()}
out={'time':time.time(),'cpu_only':True,'state_order_equal':True,'camera_order_and_pixels_equal':True,'composite_tensor_bitwise_equal':True,'canvas_shape':list(x.shape),'action_roundtrip_max_error':err,'native14_command_split_join_equal':True,'source_intrinsics':sourceK,'runtime_preset':c,'intrinsic_max_errors':errors,'intrinsics_match':all(e<1e-3 for e in errors.values()),'limitations':'CPU preprocessing parity on a recorded observation; does not prove simulator contact dynamics or historical extrinsics/assets match'}
(B/'status/flux-preprocessing-parity-audit.json').write_text(json.dumps(out,indent=2))
PY
