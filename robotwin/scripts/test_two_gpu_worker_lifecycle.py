"""Regression for the upstream Trainer's lazy build-before-train contract."""
from types import SimpleNamespace
import sys,torch
import train_two_gpu_worker as worker

events=[];sync=[]
cfg=SimpleNamespace(param_dtype='float32',global_batch=lambda world:32)
class LazyTrainer:
    def __init__(self,config):
        self.world_size=2;self.rank=0;events.append('init')
    def build(self):
        assert events==['init'];events.append('build')
        self.emas={'ema':SimpleNamespace(shadow=[torch.zeros(1)])}
        self.state=SimpleNamespace(extra={})
        self.policy=SimpleNamespace(dit=SimpleNamespace(set_requires_gradient_sync=lambda enabled:sync.append(enabled)))
        return self
    def train(self):
        assert events==['init','build'];assert self.state.extra['memory_recipe']['ema_device']=='cpu'
        self.policy.dit.set_requires_gradient_sync(False);assert sync==[True]
        events.append('train')
worker.training.Trainer=LazyTrainer
worker.training.TrainConfig.from_file=lambda *args:cfg
torch.distributed.barrier=lambda:None
torch.distributed.destroy_process_group=lambda:None
sys.argv=['worker','unused.json'];worker.main()
assert events==['init','build','train']
print('LAZY_BUILD_SINGLE_INITIALIZATION_AND_SYNC_OVERRIDE_PASS')
