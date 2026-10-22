from pathlib import Path
import hashlib,json,math,pickle,sys,time
import torch
from safetensors import safe_open
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
task=sys.argv[1];step=int(sys.argv[2]);assert task in ('cover_blocks','match_and_pick_from_conveyor') and step in (100,10000)
out=Path(str(R).replace('/hdd/','/ssd/'))/'runs'/f'{task}-10k-prefix-v1'
p=out/f'step-{step}';assert (p/'COMPLETE').exists()
state=json.loads((p/'state.json').read_text());assert state['step']==step and state['world_size']==2
components={}
for n in ('model','optimizer','ema_0p10','ema_0p05'):
    with (p/n/'.metadata').open('rb') as f:meta=pickle.load(f)
    for v in meta.storage_data.values():
        target=(p/n/v.relative_path).resolve();assert target.is_relative_to((p/n).resolve())
        assert v.offset>=0 and v.length>0 and v.offset+v.length<=target.stat().st_size
    components[n]=len(meta.state_dict_metadata)
assert components=={'model':451,'optimizer':1367,'ema_0p10':447,'ema_0p05':447},components
for rank in (0,1):assert isinstance(torch.load(p/f'rng/rank-{rank:04d}.pt',weights_only=True),dict)
records=[]
for line in (out/'metrics.jsonl').read_text().splitlines():
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    if isinstance(r,dict):records.append(r)
updates=[r for r in records if 'step_time' in r]
assert updates[-1]['step']==step and all(math.isfinite(r[k]) for r in updates for k in ('loss','grad_norm'))
result={'time':time.time(),'verified':True,'task':task,'step':step,'checkpoint_components':components,'rng_ranks':[0,1],'data_contract':state['extra']['data_contract']}
if step==10000:
    assert any(r.get('event')=='resume' and r.get('step')==100 for r in records)
    export=out/'export-10000-ema';manifest=json.loads((export/'manifest.json').read_text())
    for n,h in manifest['sha256'].items():
        with (export/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    cfg=json.loads((export/'config.json').read_text());assert cfg['prefix_memory_spec']['version']=='causal_prefix_rgb_v1' and cfg['action_dim']==14 and cfg['fps']==25
    torch.set_num_threads(4)
    with safe_open(export/'model.safetensors',framework='pt',device='cpu') as f:
        assert len(f.keys())==451
        for key in f.keys():assert torch.isfinite(f.get_tensor(key)).all(),key
    result.update(export=str(export),sha256=manifest['sha256'],export_tensors=451)
name='EARLY_CHECKPOINT_VERIFIED.json' if step==100 else 'TRAIN_COMPLETE.json'
(out/name).write_text(json.dumps(result,indent=2));print(json.dumps(result))
