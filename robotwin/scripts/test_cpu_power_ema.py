import os,pathlib,torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor,Shard
from flux_action.training.checkpoint import _dcp_save,_dcp_load
from flux_action.training.ema import PowerEMA
from cpu_power_ema import CPUPowerEMA

def main():
    cuda=os.environ.get('EMA_TEST_CUDA','0')=='1'
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank) if cuda else torch.device('cpu')
    if cuda:torch.cuda.set_device(device)
    dist.init_process_group('nccl' if cuda else 'gloo')
    mesh=init_device_mesh(device.type,(dist.get_world_size(),),mesh_dim_names=('shard',))
    original=torch.arange(256,dtype=torch.float32,device=device).reshape(16,16)/256
    tensor=distribute_tensor(original.clone(),mesh,[Shard(0)])
    reference=PowerEMA([('weight',original)],.1)
    ema=CPUPowerEMA([('weight',tensor)],.1)
    for step in range(6):
        tensor.add_(.01);original.add_(.01)
        ema.update([tensor],step);reference.update([original],step)
        torch.testing.assert_close(ema.shadow[0].full_tensor(),reference.shadow[0].cpu(),rtol=1e-6,atol=1e-7)
    assert ema.shadow[0].device.type=='cpu'
    path=pathlib.Path(__file__).resolve().parents[1]/'runs'/('cpu-ema-smoke-cuda' if cuda else 'cpu-ema-smoke-cpu')
    _dcp_save({'tensors':{'weight':ema.shadow[0]}},path)
    restored=torch.zeros_like(original,device='cpu')
    _dcp_load({'tensors':{'weight':restored}},path)
    torch.testing.assert_close(restored,reference.shadow[0].cpu(),rtol=1e-6,atol=1e-7)
    dist.barrier()
    if rank==0:print('CPU_EMA_SHARDS_UPDATE_AND_GLOBAL_RELOAD_PASS',device.type,flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
