import os,sys,json,time,torch
from buffered_cpu_ema import BufferedCPUPowerEMA
from async_cpu_ema import AsyncCPUPowerEMA
import flux_action.training.trainer as training

def main():
    mode=os.environ.get('CPU_EMA_MODE','buffered');assert mode in ['buffered','async']
    torch.set_num_threads(int(os.environ.get('CPU_THREADS','4')))
    training.PowerEMA=AsyncCPUPowerEMA if mode=='async' else BufferedCPUPowerEMA
    cfg=training.TrainConfig.from_file(sys.argv[1],sys.argv[2:]);t=training.Trainer(cfg).build()
    assert t.world_size==2 and cfg.global_batch(2)==32
    setter=t.policy.dit.set_requires_gradient_sync
    t.policy.dit.set_requires_gradient_sync=lambda enabled,*a,**kw:setter(True,*a,**kw)
    t.state.extra['memory_recipe']={'ema_device':'cpu','ema_dtype':'float32','mode':mode,'sync_each_microbatch':True,'param_dtype':cfg.param_dtype}
    if os.environ.get('BENCH_NO_SAVE')=='1':
        t._checkpoint=lambda:AsyncCPUPowerEMA.flush()
    original_update=t._train_update
    def update(*args,**kwargs):
        stop=original_update(*args,**kwargs)
        # Rank0 announces stop over NCCL, avoiding divergent filesystem observations.
        request=torch.tensor(int(t.rank==0 and (t.output_dir/'STOP_AFTER_UPDATE').exists()),device=t.device)
        torch.distributed.broadcast(request,src=0)
        if request.item():
            AsyncCPUPowerEMA.flush()
            if t._written!=t.step:t._checkpoint();t._written=t.step
            if t.rank==0:(t.output_dir/'STOPPED_SAFELY.json').write_text(json.dumps({'time':time.time(),'step':t.step,'complete_checkpoint':str(t.output_dir/f'step-{t.step}')}))
            cfg.steps=t.step
            return True
        return stop
    t._train_update=update
    print(json.dumps({'event':'EFFICIENCY_RANK_READY','rank':t.rank,'mode':mode,'cpu_threads':torch.get_num_threads(),'micro':cfg.windows_per_rank,'prefetch':cfg.prefetch_factor,'step':t.step}),flush=True)
    t.train();AsyncCPUPowerEMA.flush();torch.distributed.barrier();torch.distributed.destroy_process_group()
if __name__=='__main__':main()
