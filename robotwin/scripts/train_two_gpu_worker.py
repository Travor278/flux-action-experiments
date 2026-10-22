import json,sys,torch
from cpu_power_ema import CPUPowerEMA
import flux_action.training.trainer as training

def main():
    training.PowerEMA=CPUPowerEMA
    cfg=training.TrainConfig.from_file(sys.argv[1],sys.argv[2:])
    trainer=training.Trainer(cfg).build()
    assert trainer.world_size==2
    assert all(x.device.type=='cpu' for ema in trainer.emas.values() for x in ema.shadow)
    setter=trainer.policy.dit.set_requires_gradient_sync
    def sync_every_microbatch(enabled,*args,**kwargs):
        return setter(True,*args,**kwargs)
    trainer.policy.dit.set_requires_gradient_sync=sync_every_microbatch
    trainer.state.extra['memory_recipe']={'ema_device':'cpu','ema_dtype':'float32','sync_each_microbatch':True,'param_dtype':cfg.param_dtype}
    print(json.dumps({'event':'TWO_GPU_RANK_READY','rank':trainer.rank,'world_size':2,'global_batch':cfg.global_batch(2),'memory_recipe':trainer.state.extra['memory_recipe']}),flush=True)
    trainer.train()
    torch.distributed.barrier();torch.distributed.destroy_process_group()

if __name__=='__main__':main()
