import importlib.util,os
from pathlib import Path
import numpy as np
import pytest
import torch
P=Path(os.environ.get('FLUX_ADAPTER_PATH',str(Path(__file__).parent/'policy/model.py')))
spec=importlib.util.spec_from_file_location('rd_flux_adapter',P);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class RecordingPolicy:
 def __init__(self):self.reset()
 def reset(self):self.ticks=[]
 def select_action(self,batch):
  assert set(batch)=={'state','task',*mod.CAMERAS}
  self.ticks.append(batch)
  return batch['state']+0.125

def make():
 m=object.__new__(mod.Model);m.device=torch.device('cpu');m.info={'arm_dim':[6,6],'ee_dim':[1,1]};m.policy=RecordingPolicy();m.reset();return m
def obs(t):
 image=np.zeros((8,10,3),np.uint8);image[...,0]=17;image[...,2]=233
 return {'state':np.arange(14,dtype=np.float32)+t,'instruction':'remember the target',
         'images':{'cam_high':image,'cam_left_wrist':image,'cam_right_wrist':image},
         'hidden_object_identity':'MUST NOT ENTER POLICY','future_action':np.ones(14)}
def test_every_tick_consumed_no_privileged_state_and_reset():
 m=make()
 for t in range(37):
  m.update_obs(obs(t));actions=m.get_action();assert len(actions)==1
  assert np.allclose(actions[0]['left_arm_joint_state'],np.arange(6)+t+0.125)
 assert len(m.policy.ticks)==37
 assert [x['state'][0,0].item() for x in m.policy.ticks]==list(range(37))
 m.reset();assert not m.policy.ticks and m.pending is None
 m.update_obs(obs(0));m.get_action();assert len(m.policy.ticks)==1
def test_rgb_order_preserved():
 m=make();m.update_obs(obs(0));m.get_action();im=m.policy.ticks[0]['images.head']
 assert im.dtype==torch.uint8 and im[0,0,0,0]==17 and im[0,2,0,0]==233
def test_stale_duplicate_observation_rejected():
 m=make()
 with pytest.raises(RuntimeError):m.get_action()
 m.update_obs(obs(0))
 with pytest.raises(RuntimeError):m.update_obs(obs(1))
 m.get_action()
 with pytest.raises(RuntimeError):m.get_action()
def test_no_silent_float_image_conversion():
 o=obs(0);o['images']['cam_high']=o['images']['cam_high'].astype(np.float32)
 with pytest.raises(ValueError):make().update_obs(o)
