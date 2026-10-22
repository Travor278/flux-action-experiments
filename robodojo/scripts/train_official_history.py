"""Native FLUX training with the audited RoboDojo data and memory contracts."""
import json,os,sys,time
from pathlib import Path
import torch
import flux_action.training.trainer as training
from async_cpu_ema import AsyncCPUPowerEMA
from flux_action.training.trainer import Trainer

class SafeStop(Exception):
    pass

def main():
    torch.set_num_threads(4);training.PowerEMA=AsyncCPUPowerEMA
    cfg=training.TrainConfig.from_file(sys.argv[1],sys.argv[2:])
    t=Trainer(cfg).build()
    assert t.world_size==2 and cfg.global_batch(2)==32
    assert 'H100' in torch.cuda.get_device_name(t.device)
    assert 'prefix_memory_spec' not in cfg.policy and cfg.dataset=='index'
    assert t.policy.config.n_obs_steps==8 and t.policy.config.history_snapshots==2
    setter=t.policy.dit.set_requires_gradient_sync
    t.policy.dit.set_requires_gradient_sync=lambda enabled,*args,**kwargs:setter(True,*args,**kwargs)
    t.state.extra['memory_recipe']={'ema_device':'cpu','ema_dtype':'float32','sync_each_microbatch':True,'conditioning':'upstream_history_only'}
    original=t._train_update
    def update(*args,**kwargs):
        finished=original(*args,**kwargs)
        stop=torch.tensor(int(t.rank==0 and (t.output_dir/'STOP_AFTER_UPDATE').exists()),device=t.device)
        torch.distributed.broadcast(stop,src=0)
        if stop.item():
            AsyncCPUPowerEMA.flush()
            if t._written!=t.step:t._checkpoint();t._written=t.step
            if t.rank==0:(t.output_dir/'STOPPED_SAFELY.json').write_text(json.dumps({'time':time.time(),'step':t.step}))
            raise SafeStop()
        return finished
    t._train_update=update
    print(json.dumps({'event':'ROBODOJO_FORMAL_RANK_READY','rank':t.rank,'step':t.step,'global_batch':32,'data_contract':t._data_contract()}),flush=True)
    try:
        t.train()
    except SafeStop:
        pass
    AsyncCPUPowerEMA.flush()
    torch.distributed.barrier();torch.distributed.destroy_process_group()
if __name__=='__main__':main()
