"""Actual frozen VAE equality and isolated encoding speed, not full update speed."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,time,statistics
import torch
from flux_action.config import PolicyConfig
from flux_action.models.video_vae import load_video_vae
from flux_action.processing import prefix_memory
from robodojo_history_dataset import build_dataset
from exact_prefix_cache import ExactPrefixVAECache

R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
out=Path(os.environ['FLUX_VAE_PROBE_DIR']);torch.set_num_threads(4)
weights=R.parent/'flux_piperx_20261008/assets/flux-base/video_vae.safetensors'
vae=load_video_vae(str(weights),device='cuda',compile_model=False)
cached=ExactPrefixVAECache(vae)
results=[]
for task in ['cover_blocks','match_and_pick_from_conveyor']:
    config=json.loads((R/'configs'/f'{task}-10k-prefix-v1.json').read_text())
    cfg=PolicyConfig(**config['policy'])
    loadercfg=SimpleNamespace(**{**config,'policy':cfg.to_dict()})
    ds=build_dataset(loadercfg,seed=42,rank=0,world_size=1,num_workers=2,windows_per_rank=2,grad_accumulation=8)
    for tick in [0,10,50,100]:
        items=[ds.window_at(i,tick) for i in (0,1)]
        batch={k:torch.stack([v[k] for v in items]) if isinstance(items[0][k],torch.Tensor) else torch.tensor([v[k] for v in items]) if isinstance(items[0][k],int) else [v[k] for v in items] for k in items[0]}
        samples=[]
        with torch.no_grad():
            for attempt in range(3):
                outputs=[];durations=[]
                for encoder in [vae,cached]:
                    before_cpu=torch.get_rng_state();before_gpu=torch.cuda.get_rng_state()
                    torch.cuda.synchronize();began=time.perf_counter()
                    tokens,ids=prefix_memory.encode_prefix(encoder,batch,cfg,[0,1],torch.device('cuda'))
                    torch.cuda.synchronize();durations.append(time.perf_counter()-began)
                    assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_gpu,torch.cuda.get_rng_state())
                    outputs.append((tokens.clone(),ids.clone()))
                for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=0,atol=0)
                samples.append({'baseline_seconds':durations[0],'cache_seconds':durations[1]})
        case={'task':task,'tick':tick,'exact':True,'rng_unchanged':True,'samples':samples,'hits':cached.hits,'misses':cached.misses,'cpu_cache_bytes':cached.bytes}
        results.append(case)
        with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(case)+'\n')
        print(json.dumps(case),flush=True)
warm=[s for r in results for s in r['samples'][1:]]
record={'time':time.time(),'verified':True,'cases':results,'warm_baseline_median':statistics.median(s['baseline_seconds'] for s in warm),'warm_cache_median':statistics.median(s['cache_seconds'] for s in warm),'scope':'Exact prefix encoding only; no production change; full update speed and exact optimizer/EMA/RNG equivalence still required before adoption'}
(out/'exact-prefix-cache-verified.json').write_text(json.dumps(record,indent=2))
