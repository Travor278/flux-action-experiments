from pathlib import Path
import json,time,pickle,torch
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
records={}
for task in ['cover_blocks','match_and_pick_from_conveyor']:
    out=S/'runs'/f'{task}-10k-prefix-v1'
    stopped=json.loads((out/'STOPPED_SAFELY.json').read_text()) if (out/'STOPPED_SAFELY.json').exists() else None
    if task=='cover_blocks':assert stopped,'Cover still saving or running'
    cp=max(out.glob('step-*/COMPLETE'),key=lambda p:int(p.parent.name.split('-')[1])).parent
    state=json.loads((cp/'state.json').read_text());assert state['world_size']==2
    if stopped:assert stopped['step']==state['step']
    components={}
    for n in ('model','optimizer','ema_0p10','ema_0p05'):
        with (cp/n/'.metadata').open('rb') as f:m=pickle.load(f)
        components[n]=len(m.state_dict_metadata)
        for v in m.storage_data.values():
            p=(cp/n/v.relative_path).resolve();assert p.is_relative_to((cp/n).resolve()) and v.offset+v.length<=p.stat().st_size
    assert components=={'model':451,'optimizer':1367,'ema_0p10':447,'ema_0p05':447}
    for rank in (0,1):assert isinstance(torch.load(cp/f'rng/rank-{rank:04d}.pt',weights_only=True),dict)
    records[task]={'checkpoint':str(cp),'step':state['step'],'safe_stop':stopped,'components':components,'preserve':True}
(R/'status/official-switch-stopped-verified.json').write_text(json.dumps({'time':time.time(),'verified':True,'records':records},indent=2))
print(json.dumps(records),flush=True)
