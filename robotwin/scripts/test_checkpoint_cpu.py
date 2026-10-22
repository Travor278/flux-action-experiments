import json,os,pathlib,torch
from flux_action.training.checkpoint import save_checkpoint,load_checkpoint,TrainState,ema_folder,_checkpoint_group
from flux_action.training.ema import PowerEMA
from flux_action.config import PolicyConfig
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
D=torch.distributed;D.init_process_group('gloo');rank=D.get_rank();torch.manual_seed(42)
m=torch.nn.Linear(4,3);o=torch.optim.AdamW(m.parameters(),lr=.001)
(m(torch.ones(2,4)).square().mean()).backward();o.step();o.zero_grad()
emas={ema_folder(s):PowerEMA.from_module(m,s) for s in [.1,.05]}
for e in emas.values():e.update(list(m.parameters()),1)
expected=[p.detach().clone() for p in m.parameters()]
expected_o=get_optimizer_state_dict(m,o)
root=pathlib.Path(__file__).resolve().parents[1]/'runs/checkpoint-cpu-smoke'
p=save_checkpoint(root,step=1,model=m,optimizer=o,emas=emas,state=TrainState(world_size=D.get_world_size()),config=PolicyConfig())
assert D.get_backend(_checkpoint_group())=='gloo'
for x in m.parameters():x.data.zero_()
for e in emas.values():
 for x in e.shadow:x.zero_()
s=load_checkpoint(p,model=m,optimizer=o,emas=emas)
assert s.step==1 and (p/'COMPLETE').exists()
for a,b in zip(m.parameters(),expected):assert torch.equal(a,b)
for e in emas.values():
 for a,b in zip(e.shadow,expected):assert torch.equal(a,b)
actual_o=get_optimizer_state_dict(m,o)
for name,d in expected_o['state'].items():
 for k,v in d.items():
  if torch.is_tensor(v):assert torch.equal(v,actual_o['state'][name][k])
D.barrier()
if rank==0:print('CPU_2RANK_MODEL_OPTIMIZER_DUAL_EMA_SAVE_RESTORE_PASS',flush=True)
D.destroy_process_group()
