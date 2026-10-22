"""AdamW with CPU-resident moments and the same per-tensor CUDA fused update."""
import torch

class CPUStateAdamW(torch.optim.AdamW):
    def __init__(self,*args,**kwargs):
        kwargs['fused']=True;kwargs['foreach']=False
        super().__init__(*args,**kwargs)

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        groups=self.param_groups
        try:
            for group in groups:
                for param in group['params']:
                    if param.grad is None:continue
                    state=self.state[param]
                    for key,value in list(state.items()):
                        if torch.is_tensor(value):state[key]=value.to(param.device)
                    self.param_groups=[{**group,'params':[param]}]
                    super().step()
                    for key,value in list(state.items()):
                        if torch.is_tensor(value):state[key]=value.to('cpu')
        finally:self.param_groups=groups
        return loss

    def load_state_dict(self,state_dict):
        # Small-state roundtrips work; production resume needs a CPU-first loader
        # to avoid transiently staging every moment on the 32GB device.
        super().load_state_dict(state_dict)
        for state in self.state.values():
            for key,value in list(state.items()):
                if torch.is_tensor(value):state[key]=value.to('cpu')
