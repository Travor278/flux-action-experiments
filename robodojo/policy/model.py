"""XPolicyLab adapter for a RoboDojo-trained native FLUX history export.

One policy action per simulator control tick. FLUX owns its action queue and
records every observation, including queued ticks. No privileged task fields.
GPU/checkpoint/simulator integration is a required later acceptance gate.
"""
import numpy as np
import torch
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

CAMERAS={
 'images.head': ('cam_high','cam_head','head_camera','top_camera'),
 'images.left_wrist': ('cam_left_wrist','left_camera','left_wrist','wrist_left'),
 'images.right_wrist': ('cam_right_wrist','right_camera','right_wrist','wrist_right'),
}

def image_chw(obs,names):
 vision=obs.get('images',obs.get('vision',{}))
 for name in names:
  if name not in vision:continue
  image=vision[name]
  if isinstance(image,dict):image=image.get('color',image.get('rgb'))
  a=np.asarray(image)
  if a.dtype!=np.uint8 or a.ndim!=3:raise ValueError('Require decoded uint8 RGB; do not silently rescale or swap channels')
  if a.shape[-1]==3:a=a.transpose(2,0,1)
  elif a.shape[0]!=3:raise ValueError('Require three RGB channels')
  return np.ascontiguousarray(a)
 raise KeyError(names)

class Model(ModelTemplate):
 def __init__(self,model_cfg):
  from flux_action.policy import FluxActionPolicy
  if model_cfg.get('action_type')!='joint' or model_cfg.get('env_cfg_type')!='arx_x5':raise ValueError('This adapter is ARX X5 joint-only')
  self.info=get_robot_action_dim_info('arx_x5')
  self.device=torch.device(model_cfg.get('device','cuda'))
  self.policy=FluxActionPolicy.from_pretrained(model_cfg['ckpt_name'],device=self.device)
  cfg=self.policy.config
  assert cfg.action_dim==sum(self.info['arm_dim'])+sum(self.info['ee_dim'])==14
  assert cfg.inference_profile=='history' and cfg.n_obs_steps>1 and cfg.condition_on_past_actions
  assert cfg.fps==25 and tuple(cfg.camera_keys)==tuple(CAMERAS)
  self.reset()
 def reset(self):
  self.policy.reset();self.pending=None
 def update_obs(self,obs):
  if self.pending is not None:raise RuntimeError('Previous observation was never consumed')
  prompt=obs.get('instruction')
  if not isinstance(prompt,str) or not prompt:raise ValueError('Literal nonempty instruction required')
  state=np.asarray(obs['state'],dtype=np.float32) if not isinstance(obs['state'],dict) else pack_robot_state(obs,'joint',self.info).astype(np.float32)
  if state.shape!=(14,) or not np.isfinite(state).all():raise ValueError('Invalid measured state')
  self.pending={'state':torch.from_numpy(state.copy())[None].to(self.device),'task':[prompt]}
  self.pending.update({key:torch.from_numpy(image_chw(obs,names))[None].to(self.device) for key,names in CAMERAS.items()})
 @torch.inference_mode()
 def get_action(self):
  if self.pending is None:raise RuntimeError('Each action needs a fresh control-tick observation')
  batch=self.pending;self.pending=None
  a=self.policy.select_action(batch).float().cpu().numpy()
  if a.shape!=(1,14) or not np.isfinite(a).all():raise ValueError('Invalid policy action')
  return unpack_robot_state(a,'joint',self.info,source_type='obs')
 def update_obs_batch(self,*args,**kwargs):raise NotImplementedError('One environment per policy; prevent cross-episode history leakage')
 def get_action_batch(self,*args,**kwargs):raise NotImplementedError('Single environment only')
