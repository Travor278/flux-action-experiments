"""Same-checkpoint full-update comparison; outputs are diagnostic fingerprints only."""
import functools,gzip,json,os,sys,time
from pathlib import Path
import torch
import flux_action.training.trainer as training
from flux_action.processing import prefix_memory
from async_cpu_ema import AsyncCPUPowerEMA
from robodojo_training_contract import RoboDojoTrainer
from train_full_preflight import recovery_fingerprint
from exact_prefix_cache import ExactPrefixVAECache

def main():
    mode=sys.argv[2];assert mode in ('baseline','cached')
    cfg=training.TrainConfig.from_file(sys.argv[1],[])
    original=json.loads(Path(os.environ['PERF_ORIGINAL_CONFIG']).read_text())
    current=json.loads(Path(sys.argv[1]).read_text())
    allowed={'output_dir','resume','steps','checkpoint_every','log_every','keep_checkpoints'}
    assert all(current[k]==v for k,v in original.items() if k not in allowed)
    torch.set_num_threads(4);training.PowerEMA=AsyncCPUPowerEMA
    t=RoboDojoTrainer(cfg).build();start=t.step
    assert t.world_size==2 and cfg.global_batch(2)==32 and cfg.windows_per_rank==2 and cfg.grad_accumulation==8
    assert t.state.extra['resumes'][-1]['mode']=='exact'
    assert 'H100' in torch.cuda.get_device_name(t.device)
    t.output_dir.mkdir(parents=True,exist_ok=True)
    fingerprint_before=recovery_fingerprint(t)
    cache=ExactPrefixVAECache(t.policy.video_vae)
    native_prefix=prefix_memory.encode_prefix
    def encode(vae,*a,**kw):
        with torch.profiler.record_function('frozen_prefix_encode'):
            return native_prefix(cache if mode=='cached' else vae,*a,**kw)
    prefix_memory.encode_prefix=encode
    sync=t.policy.dit.set_requires_gradient_sync
    t.policy.dit.set_requires_gradient_sync=lambda enabled,*a,**kw:sync(True,*a,**kw)
    timings={};events=[]
    def instrument(obj,name,label):
        fn=getattr(obj,name)
        @functools.wraps(fn)
        def call(*a,**kw):
            began=time.perf_counter();first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
            first.record()
            with torch.profiler.record_function(label):result=fn(*a,**kw)
            last.record();events.append((label,first,last))
            timings.setdefault(label,[]).append(time.perf_counter()-began)
            return result
        setattr(obj,name,call)
    instrument(t.policy,'prepare','prepare_all_encoders')
    instrument(t.policy,'forward','policy_forward')
    instrument(t.optimizer,'step','optimizer_step')
    for key,ema in t.emas.items():instrument(ema,'update_from_module',key)
    # No diagnostic checkpoint/optimizer files: retain original recovery state.
    t._checkpoint=lambda:None
    native_update=t._update
    reports=[]
    def update(*a,**kw):
        timings.clear();events.clear()
        profile=t.step==start+4
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
                result=native_update(*a,**kw)
            trace=t.output_dir/f'rank{t.rank}-trace.json'
            prof.export_chrome_trace(str(trace))
            with trace.open('rb') as src,gzip.open(str(trace)+'.gz','wb') as dst:
                import shutil;shutil.copyfileobj(src,dst)
            trace.unlink()
            (t.output_dir/f'rank{t.rank}-operators.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total',row_limit=35))
        else:result=native_update(*a,**kw)
        gpu={}
        for label,first,last in events:gpu[label]=gpu.get(label,0)+first.elapsed_time(last)/1000
        reports.append({'step':t.step+1,'profiled':profile,'metrics':result,'cpu_host_seconds':{k:sum(v) for k,v in timings.items()},'gpu_stream_intervals':gpu,'cache_hits':cache.hits,'cache_misses':cache.misses,'cache_bytes':cache.bytes})
        return result
    t._update=update
    t.train();AsyncCPUPowerEMA.flush()
    record={'time':time.time(),'mode':mode,'rank':t.rank,'start':start,'end':t.step,'before':fingerprint_before,'after':recovery_fingerprint(t),'updates':reports,'no_production_mutation':True}
    (t.output_dir/f'comparison-rank{t.rank}.json').write_text(json.dumps(record,indent=2))
    torch.distributed.barrier();torch.distributed.destroy_process_group()
if __name__=='__main__':main()
