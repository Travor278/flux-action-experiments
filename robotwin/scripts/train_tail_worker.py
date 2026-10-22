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
    if t.step == 0:
        from robotwin_tail_dataset import build
        from flux_action.training.data import collate_windows
        ds=build(cfg,seed=42,epoch=0,rank=t.rank,world_size=2,num_workers=2,windows_per_rank=4,skip_batches=0,grad_accumulation=4)
        examples=[ds.window_at(0,ds.episodes[0]['n_frames']-1),ds.window_at(49,ds.episodes[49]['n_frames']-5)]
        with torch.random.fork_rng(devices=[t.device]),torch.no_grad():
            batch=collate_windows(examples); prepared=t.policy.prepare(batch)
            assert len(prepared.idx)==2 and prepared.action_time_mask.sum(-1).tolist()==[1,5]
            assert prepared.video_time_mask.sum(-1).tolist()==[0,prepared.latents.shape[-2]*prepared.latents.shape[-1]]
            loss,info=t.policy(batch,prepared=prepared)
            assert torch.isfinite(loss) and info['n_valid_windows']==2
            print(json.dumps({'event':'TAIL_GPU_PREFLIGHT_PASS','rank':t.rank,'real_action_counts':[1,5],'loss':float(loss),'video_valid_tokens':prepared.video_time_mask.sum(-1).tolist()}),flush=True)
        del prepared,batch,examples,ds,loss,info
        torch.cuda.empty_cache()
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
