import sys,time,json,os,torch
from cpu_power_ema import CPUPowerEMA
from buffered_cpu_ema import BufferedCPUPowerEMA
import flux_action.training.trainer as training

def main():
    training.PowerEMA=BufferedCPUPowerEMA if os.environ.get('PERF_BUFFERED')=='1' else CPUPowerEMA
    cfg=training.TrainConfig.from_file(sys.argv[1],sys.argv[2:]);t=training.Trainer(cfg).build()
    assert t.world_size==2 and cfg.global_batch(2)==32
    sync=t.policy.dit.set_requires_gradient_sync
    t.policy.dit.set_requires_gradient_sync=lambda enabled,*a,**kw:sync(True,*a,**kw)
    totals={};original_update=t._update
    def timed(name,fn):
        def run(*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter();result=fn(*args,**kwargs);torch.cuda.synchronize()
            totals[name]=totals.get(name,0)+time.perf_counter()-start
            return result
        return run
    t.optimizer.step=timed('optimizer',t.optimizer.step)
    for name,e in t.emas.items():e.update_from_module=timed(name,e.update_from_module)
    def update(*args,**kwargs):
        if t.step==5:
            for e in t.emas.values():e.__class__=BufferedCPUPowerEMA
        totals.clear();torch.cuda.synchronize();start=time.perf_counter();result=original_update(*args,**kwargs)
        total=time.perf_counter()-start
        mode='buffered' if isinstance(next(iter(t.emas.values())),BufferedCPUPowerEMA) else 'baseline'
        record={'time':time.time(),'rank':t.rank,'step':t.step+1,'mode':mode,'update_seconds':total,'phases':dict(totals),'compute_and_other_seconds':total-sum(totals.values()),'cpu_threads':torch.get_num_threads()}
        with open(t.output_dir/f'phases-rank{t.rank}.jsonl','a') as f:f.write(json.dumps(record)+'\n')
        print('PERF_PHASE '+json.dumps(record),flush=True)
        return result
    t._update=update
    print(json.dumps({'event':'PERF_RANK_READY','rank':t.rank,'start_step':t.step}),flush=True)
    t.train();torch.distributed.barrier();torch.distributed.destroy_process_group()
if __name__=='__main__':main()
