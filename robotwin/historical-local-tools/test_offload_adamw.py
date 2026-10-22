import copy,json,pathlib,torch
from offload_adamw import CPUStateAdamW
torch.manual_seed(7)
for dtype in [torch.float32,torch.bfloat16]:
 a=[torch.nn.Parameter(torch.randn(17,29,device='cuda',dtype=dtype)) for _ in range(4)]
 b=[torch.nn.Parameter(x.detach().clone()) for x in a]
 kw=dict(lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
 ref=torch.optim.AdamW([{'params':a[:2]},{'params':a[2:],'lr':5e-5}],fused=True,foreach=False,**kw)
 off=CPUStateAdamW([{'params':b[:2]},{'params':b[2:],'lr':5e-5}],**kw)
 for step in range(15):
  for x,y in zip(a,b):x.grad=torch.randn_like(x);y.grad=x.grad.clone()
  ref.step();off.step()
  for x,y in zip(a,b):
   assert torch.equal(x,y),('parameter mismatch',dtype,step)
   for key,v in ref.state[x].items():
    assert off.state[y][key].device.type=='cpu'
    assert torch.equal(v.cpu(),off.state[y][key]),('state mismatch',key,dtype,step)
  if step==7:off.load_state_dict(copy.deepcopy(off.state_dict()))
print('CPU_STATE_FUSED_ADAMW_EXACT_FP32_BF16_15_STEPS_PASS',flush=True)
pathlib.Path(__file__).resolve().parents[1].joinpath('status/offload-adamw-verified.json').write_text(json.dumps({'passed':True,'steps_each':15,'dtypes':['float32','bfloat16'],'multi_group':True,'small_state_roundtrip':True}))
