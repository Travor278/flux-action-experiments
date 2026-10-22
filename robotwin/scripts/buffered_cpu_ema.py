"""One synchronized D2H snapshot per optimizer update, shared by both CPU EMAs."""
import torch
from torch.distributed.tensor import DTensor
from cpu_power_ema import CPUPowerEMA
from flux_action.training.ema import power_ema_beta

class BufferedCPUPowerEMA(CPUPowerEMA):
    _buffers = None
    _snapshot_key = None
    _snapshot_step = None

    @torch.no_grad()
    def update(self, tensors, step):
        sources=[p.to_local() if isinstance(p,DTensor) else p for p in tensors]
        key=tuple((p.data_ptr(),tuple(p.shape),p.dtype,p.device) for p in sources)
        cls=BufferedCPUPowerEMA
        if cls._snapshot_key!=key:
            cls._buffers=[torch.empty_like(p,device='cpu',dtype=torch.float32,pin_memory=p.device.type=='cuda') for p in sources]
            cls._snapshot_key=key;cls._snapshot_step=None
        if cls._snapshot_step!=step:
            for dst,src in zip(cls._buffers,sources,strict=True):dst.copy_(src.detach(),non_blocking=src.device.type=='cuda')
            if sources[0].device.type=='cuda':torch.cuda.current_stream(sources[0].device).synchronize()
            cls._snapshot_step=step
        beta=power_ema_beta(self.gamma,step)
        for target,value in zip(self.shadow,cls._buffers,strict=True):
            target=target.to_local() if isinstance(target,DTensor) else target
            if step==0:target.copy_(value)
            else:target.lerp_(value,1-beta)
        self.last_step=step
        return beta
