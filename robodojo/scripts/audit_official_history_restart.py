"""Validate stopped production checkpoints for the user's explicit restart."""
from pathlib import Path
import hashlib,json,math,pickle,time,torch

from flux_action.training.trainer import TrainConfig
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
gate=json.loads((R/'status/official-history-ready.json').read_text())
for name,h in gate['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
tasks={}
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 cfg=TrainConfig.from_file(str(R/'configs'/f'{task}-10k-official-history-v1.json'),[])
 assert cfg.global_batch(2)==32 and cfg.windows_per_rank==2 and cfg.grad_accumulation==8
 out=Path(cfg.output_dir).resolve();assert out.is_relative_to((S/'runs').resolve())
 assert not (out/'TRAIN_COMPLETE.json').exists()
 checkpoints=sorted(out.glob('step-*/COMPLETE'),key=lambda p:int(p.parent.name.split('-')[1]));assert checkpoints
 p=checkpoints[-1].parent;state=json.loads((p/'state.json').read_text());step=int(p.name.split('-')[1])
 assert state['step']==step and state['world_size']==2 and state['extra']['global_batch']==32
 assert state['extra']['data_contract']==hashlib.sha256(json.dumps({'dataset':cfg.dataset,'manifest':json.loads((Path(cfg.index_dir)/'manifest.json').read_text())},sort_keys=True).encode()).hexdigest()
 components={}
 for name in ('model','optimizer','ema_0p10','ema_0p05'):
  with (p/name/'.metadata').open('rb') as f:metadata=pickle.load(f)
  components[name]=len(metadata.state_dict_metadata)
  for value in metadata.storage_data.values():
   target=(p/name/value.relative_path).resolve()
   assert target.is_relative_to((p/name).resolve()) and value.offset>=0 and value.length>0 and value.offset+value.length<=target.stat().st_size
 assert components=={'model':451,'optimizer':1367,'ema_0p10':447,'ema_0p05':447}
 for rank in (0,1):assert isinstance(torch.load(p/f'rng/rank-{rank:04d}.pt',map_location='cpu',weights_only=True),dict)
 rows=[]
 for line in (out/'metrics.jsonl').read_text().splitlines():
  try:x=json.loads(line)
  except ValueError:continue
  if isinstance(x,dict) and 'step_time' in x:rows.append(x)
 assert rows and all(math.isfinite(x[k]) for x in rows for k in ['loss','grad_norm'])
 flags=[name for name in ['STOP_AFTER_UPDATE','STOPPED_SAFELY.json'] if (out/name).exists()]
 assert not flags,('Archive only after inspecting unexpected stop flags',task,flags)
 tasks[task]={'checkpoint':str(p),'resume_step':step,'last_logged_step':rows[-1]['step'],'last_logged_time':rows[-1]['time'],'last_loss':rows[-1]['loss'],'last_grad':rows[-1]['grad_norm'],'components':components,'two_rng':True,'data_contract':state['extra']['data_contract'],'data_position':state['data']['position'],'config_sha256':hashlib.sha256((R/'configs'/f'{task}-10k-official-history-v1.json').read_bytes()).hexdigest()}
audit={'time':time.time(),'verified':True,'user_requested_restart':True,'tasks':tasks,'scope':'Restore latest complete checkpoint; later unsaved updates cannot be recovered from logs'}
(R/'status/official-history-restart-audit-20261015.json').write_text(json.dumps(audit,indent=2))
gate.update(time=time.time(),restart_audit=audit,scope='User-authorized official history exact resume; no new method or hyperparameter changes')
(R/'status/official-history-restart-ready-20261015.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(audit),flush=True)
