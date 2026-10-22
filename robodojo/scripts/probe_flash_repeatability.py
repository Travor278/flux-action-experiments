"""Small same-input flash-backward repeatability test, not a training change."""
from pathlib import Path
import json,time,hashlib
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
import torch.nn.functional as F
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
torch.set_num_threads(1);torch.cuda.set_device(0)
free,total=torch.cuda.mem_get_info()
if free<1536*1024**2:
 (R/'status/flash-repeatability.json').write_text(json.dumps({'time':time.time(),'skipped':True,'reason':'Less than 1.5GiB free; preserve active production headroom','free_bytes':free}))
 raise SystemExit(0)
torch.manual_seed(9811)
torch.cuda.reset_peak_memory_stats();results=[]
for length in [256,1024]:
 q,k,v=[torch.randn((1,4,length,128),device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
 dy=torch.randn_like(q);reference=None;rows=[];rng=torch.cuda.get_rng_state()
 for repeat in range(8):
  with sdpa_kernel(SDPBackend.FLASH_ATTENTION):y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
  grads=torch.autograd.grad(y,(q,k,v),dy)
  current=[y.detach().clone(),*(g.detach().clone() for g in grads)]
  if reference is None:reference=current
  rows.append({'repeat':repeat,'forward_exact':torch.equal(reference[0],current[0]),'gradient_exact':[torch.equal(a,b) for a,b in zip(reference[1:],current[1:])],'gradient_max_abs':[float((a.float()-b.float()).abs().max()) for a,b in zip(reference[1:],current[1:])]})
  assert torch.equal(rng,torch.cuda.get_rng_state())
 results.append({'length':length,'rows':rows})
 del q,k,v,dy,y,grads,current,reference
torch.cuda.synchronize()
(R/'status/flash-repeatability.json').write_text(json.dumps({'time':time.time(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'default_deterministic':torch.are_deterministic_algorithms_enabled(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cases':results,'scope':'Small repeated identical inputs on PyTorch flash SDPA, same backend observed in full profile. Does not establish that all full-model differences come from this kernel; no production settings changed.'},indent=2))
