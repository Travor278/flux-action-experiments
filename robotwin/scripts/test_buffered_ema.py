import torch
from cpu_power_ema import CPUPowerEMA
from buffered_cpu_ema import BufferedCPUPowerEMA
torch.manual_seed(71)
p=[torch.randn(64,128,device='cuda'),torch.randn(70,device='cuda')]
old=[CPUPowerEMA(list(zip(['a','b'],p)),s) for s in [.1,.05]]
new=[BufferedCPUPowerEMA(list(zip(['a','b'],p)),s) for s in [.1,.05]]
for step in range(10):
 for x in p:x.add_(torch.randn_like(x)*.1)
 for a,b in zip(old,new):
  assert a.update(p,step)==b.update(p,step)
  for x,y in zip(a.shadow,b.shadow):torch.testing.assert_close(x,y,rtol=0,atol=0)
print('BUFFERED_TWO_EMA_EXACT_EQUIVALENCE_PASS',flush=True)
