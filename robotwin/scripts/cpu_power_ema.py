"""CPU-resident FP32 EMA shards with full DCP global-shape metadata."""
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from flux_action.training.ema import PowerEMA,power_ema_beta,sigma_rel_to_gamma

_MESHES={}
def cpu_shadow(tensor):
    if isinstance(tensor,DTensor):
        source=tensor.device_mesh
        key=(tuple(source.mesh.shape),tuple(source.mesh.flatten().tolist()),source.mesh_dim_names)
        if key not in _MESHES:
            # A 1-D mesh covering WORLD otherwise reuses the default NCCL group,
            # even when its device_type is 'cpu'. Bind an explicit Gloo group.
            assert source.mesh.ndim==1,'This recipe is validated for one-dimensional full sharding'
            group=torch.distributed.new_group(ranks=source.mesh.flatten().tolist(),backend='gloo')
            _MESHES[key]=DeviceMesh.from_group(group,'cpu',mesh=source.mesh.cpu(),mesh_dim_names=source.mesh_dim_names)
        local=tensor.to_local().detach().to(device='cpu',dtype=torch.float32,copy=True)
        return DTensor.from_local(local,_MESHES[key],tensor.placements,run_check=False,
                                  shape=tensor.shape,stride=tensor.stride())
    return tensor.detach().to(device='cpu',dtype=torch.float32,copy=True)

class CPUPowerEMA(PowerEMA):
    def __init__(self,named_tensors,sigma_rel):
        self.sigma_rel=float(sigma_rel);self.gamma=sigma_rel_to_gamma(sigma_rel)
        items=list(named_tensors);assert items
        self.names=tuple(name for name,_ in items)
        self.shadow=[cpu_shadow(t) for _,t in items];self.last_step=None

    @torch.no_grad()
    def update(self,tensors,step):
        assert len(tensors)==len(self.shadow)
        beta=power_ema_beta(self.gamma,step)
        for target,source in zip(self.shadow,tensors,strict=True):
            target=target.to_local() if isinstance(target,DTensor) else target
            source=source.to_local() if isinstance(source,DTensor) else source
            value=source.detach().to(device='cpu',dtype=torch.float32)
            if step==0:target.copy_(value)
            else:target.lerp_(value,1-beta)
        self.last_step=step
        return beta
