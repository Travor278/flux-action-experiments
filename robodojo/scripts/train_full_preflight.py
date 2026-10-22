"""Two-H100 diagnostic updates, CPU FP32 dual EMA, exact checkpoint recovery."""
import json,os,sys,time,hashlib
from pathlib import Path
import torch
from async_cpu_ema import AsyncCPUPowerEMA
import flux_action.training.trainer as training

def recovery_fingerprint(t):
    from torch.distributed.tensor import DTensor
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
    from flux_action.training.checkpoint import rng_state
    AsyncCPUPowerEMA.flush()
    def fingerprint(value):
        h=hashlib.sha256()
        def walk(x):
            if isinstance(x,DTensor):x=x.to_local()
            if torch.is_tensor(x):
                y=x.detach().cpu().contiguous();h.update(str((tuple(y.shape),str(y.dtype))).encode());h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(x,dict):
                for k in sorted(x,key=str):h.update(str(k).encode());walk(x[k])
            elif isinstance(x,(list,tuple)):
                for v in x:walk(v)
            else:h.update(repr(x).encode())
        walk(value);return h.hexdigest()
    return {'model':fingerprint(dict(t.policy.dit.named_parameters())),
            'optimizer':fingerprint(get_optimizer_state_dict(t.policy,t.optimizer)),
            'emas':fingerprint({k:v.state_dict() for k,v in t.emas.items()}),
            'rng':fingerprint(rng_state()),'data_position':t.position.to_dict()}

def main():
    torch.set_num_threads(4)
    training.PowerEMA=AsyncCPUPowerEMA
    cfg=training.TrainConfig.from_file(sys.argv[1],sys.argv[2:])
    t=training.Trainer(cfg).build()
    assert t.world_size==2 and cfg.global_batch(2)==32
    assert all('H100' in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    assert cfg.policy['prefix_memory_spec']['version']=='causal_prefix_rgb_v1'
    receipt=t.output_dir/f'exact-state-step{t.step}-rank{t.rank}.json'
    if t.step>0 and receipt.exists():
        actual=recovery_fingerprint(t);expected=json.loads(receipt.read_text())
        assert actual==expected, {'rank':t.rank,'mismatched':[k for k in actual if actual[k]!=expected[k]]}
        proof={'time':time.time(),'verified':True,'step':t.step,'fingerprints':actual}
        (t.output_dir/f'exact-resume-step{t.step}-rank{t.rank}.json').write_text(json.dumps(proof,indent=2))
        (t.output_dir/f'exact-resume-rank{t.rank}.json').write_text(json.dumps(proof,indent=2))
    elif t.step>0:
        # Attempt1 saved step2 successfully but its optional fingerprint helper
        # imported a nonexistent name. Preserve that checkpoint; never fabricate
        # the missing pre-save in-memory hash. The next save3/restore3 will provide
        # a genuine before/after comparison in a separate process.
        assert t.step==2 and os.environ.get('RECOVER_UNFINGERPRINTED_STEP2')=='1'
        r=json.loads((Path(__file__).resolve().parents[1]/'status/full-preflight-step2-independent-audit.json').read_text())
        assert r['verified'] and r['step']==2
        saved=json.loads((t.output_dir/'step-2/state.json').read_text())
        assert saved['data']['position']==t.position.to_dict()
        from flux_action.training.checkpoint import rng_state
        loaded=torch.load(t.output_dir/f'step-2/rng/rank-{t.rank:04d}.pt',weights_only=True)
        def same(a,b):
            if torch.is_tensor(a):return torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
            return a==b
        assert same(rng_state(),loaded)
        print(json.dumps({'event':'STEP2_STANDARD_RESTORE','rank':t.rank,'step':2,'rng_and_data_exact':True,'presave_model_hash_available':False}),flush=True)
    if t.step==0:
        from robodojo_history_dataset import build_dataset
        from flux_action.training.data import collate_windows
        ds=build_dataset(cfg,seed=42,epoch=0,rank=t.rank,world_size=2,num_workers=2,windows_per_rank=2,skip_batches=0,grad_accumulation=8)
        examples=[ds.window_at(0,ds.episodes[0]['n_frames']-1),ds.window_at(24,ds.episodes[24]['n_frames']-6)]
        with torch.random.fork_rng(devices=[t.device]),torch.no_grad():
            batch=collate_windows(examples);prepared=t.policy.prepare(batch)
            assert prepared.action_time_mask.sum(1).tolist()==[1,6]
            assert prepared.prefix_tokens is not None and prepared.prefix_tokens.shape[0]==2
            loss,info=t.policy(batch,prepared=prepared)
            assert torch.isfinite(loss) and info['n_valid_windows']==2
            print(json.dumps({'event':'FULL_PREFIX_TAIL_PREFLIGHT_PASS','rank':t.rank,'loss':float(loss),'valid_actions':[1,6],'prefix_tokens':list(prepared.prefix_tokens.shape)}),flush=True)
        del ds,examples,batch,prepared,loss,info
        torch.cuda.empty_cache()
    setter=t.policy.dit.set_requires_gradient_sync
    t.policy.dit.set_requires_gradient_sync=lambda enabled,*a,**kw:setter(True,*a,**kw)
    t.state.extra['memory_recipe']={'ema_device':'cpu','ema_dtype':'float32','sync_each_microbatch':True,'prefix_memory_spec':cfg.policy['prefix_memory_spec']}
    original=t._train_update
    def update(*args,**kwargs):
        done=original(*args,**kwargs)
        stop=torch.tensor(int(t.rank==0 and (t.output_dir/'STOP_AFTER_UPDATE').exists()),device=t.device)
        torch.distributed.broadcast(stop,src=0)
        if stop.item():
            AsyncCPUPowerEMA.flush()
            if t._written!=t.step:t._checkpoint();t._written=t.step
            if t.rank==0:(t.output_dir/'STOPPED_SAFELY.json').write_text(json.dumps({'time':time.time(),'step':t.step}))
            return True
        return done
    t._train_update=update
    print(json.dumps({'event':'FULL_PREFLIGHT_RANK_READY','rank':t.rank,'step':t.step,'micro':cfg.windows_per_rank,'global_batch':32}),flush=True)
    t.train();AsyncCPUPowerEMA.flush()
    (t.output_dir/f'exact-state-step{t.step}-rank{t.rank}.json').write_text(json.dumps(recovery_fingerprint(t),indent=2))
    torch.distributed.barrier();torch.distributed.destroy_process_group()

if __name__=='__main__':main()
