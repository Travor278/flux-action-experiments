"""CPU-only EMA arithmetic overlaps the next update; flush before checkpoint access."""
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.distributed.tensor import DTensor
from cpu_power_ema import CPUPowerEMA
from flux_action.training.ema import power_ema_beta

class AsyncCPUPowerEMA(CPUPowerEMA):
    _pool=None
    _pending=[]
    _buffers=None
    _key=None
    _step=None

    @classmethod
    def flush(cls):
        pending,cls._pending=cls._pending,[]
        for future in pending:future.result()

    @staticmethod
    @torch.no_grad()
    def _apply(targets,values,beta,step):
        for target,value in zip(targets,values,strict=True):
            if step==0:target.copy_(value)
            else:target.lerp_(value,1-beta)

    @torch.no_grad()
    def update(self,tensors,step):
        cls=AsyncCPUPowerEMA
        sources=[p.to_local() if isinstance(p,DTensor) else p for p in tensors]
        key=tuple((p.data_ptr(),tuple(p.shape),p.dtype,p.device) for p in sources)
        if cls._key!=key or cls._step!=step:
            cls.flush()  # Never overwrite a staging buffer still used by the CPU worker.
            if cls._key!=key:
                cls._buffers=[torch.empty_like(p,device='cpu',dtype=torch.float32,pin_memory=p.device.type=='cuda') for p in sources]
                cls._key=key
            for dst,src in zip(cls._buffers,sources,strict=True):dst.copy_(src.detach(),non_blocking=src.device.type=='cuda')
            if sources[0].device.type=='cuda':torch.cuda.current_stream(sources[0].device).synchronize()
            cls._step=step
        if cls._pool is None:cls._pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='flux-cpu-ema')
        targets=[p.to_local() if isinstance(p,DTensor) else p for p in self.shadow]
        beta=power_ema_beta(self.gamma,step)
        cls._pending.append(cls._pool.submit(cls._apply,targets,cls._buffers,beta,step))
        self.last_step=step
        return beta

    def state_dict(self,*args,**kwargs):
        self.flush()
        return super().state_dict(*args,**kwargs)

    def load_state_dict(self,*args,**kwargs):
        self.flush()
        AsyncCPUPowerEMA._step=None
        return super().load_state_dict(*args,**kwargs)

    def copy_to(self,*args,**kwargs):
        self.flush()
        return super().copy_to(*args,**kwargs)

    def consolidated(self,*args,**kwargs):
        self.flush()
        return super().consolidated(*args,**kwargs)
