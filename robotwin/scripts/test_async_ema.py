import torch
from cpu_power_ema import CPUPowerEMA
from async_cpu_ema import AsyncCPUPowerEMA
torch.manual_seed(79)
p=[torch.randn(256,512,device='cuda'),torch.randn(71,device='cuda')]
old=[CPUPowerEMA(list(zip(['a','b'],p)),s) for s in [.1,.05]]
new=[AsyncCPUPowerEMA(list(zip(['a','b'],p)),s) for s in [.1,.05]]
for step in range(20):
 for x in p:x.add_(torch.randn_like(x)*.1)
 for a,b in zip(old,new):assert a.update(p,step)==b.update(p,step)
 # Continue GPU work and mutate model before waiting; snapshot must remain independent.
 for x in p:x.add_(.01)
 if step in [0,7,19]:
  new[0].state_dict()  # Checkpoint entry must drain BOTH profiles, not only itself.
  for a,b in zip(old,new):
   for x,y in zip(a.shadow,b.shadow):torch.testing.assert_close(x,y,rtol=0,atol=0)
AsyncCPUPowerEMA.flush()
print('ASYNC_DUAL_EMA_EXACT_SNAPSHOT_AND_CHECKPOINT_FLUSH_PASS',flush=True)
