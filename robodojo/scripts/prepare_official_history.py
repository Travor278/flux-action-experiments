"""Use pristine upstream FLUX and its native WindowDataset; no memory extensions."""
from pathlib import Path
import copy,hashlib,json,os,tarfile,time
import numpy as np
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
S=Path(str(R).replace('/hdd/','/ssd/'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
archive=R/'official-flux-830f1ce.tar.gz'
assert sha(archive)=='f8527d3ba110e3b2b1a2551eedb62968ecfb54847292859ba5f6bbaa3e69ceb1'
code=R/'flux-action-official-v1'
code.mkdir(exist_ok=True)
with tarfile.open(archive) as f:
    for member in f.getmembers():
        assert (code/member.name).resolve().is_relative_to(code.resolve())
    f.extractall(code,filter='data')
import sys
sys.path.insert(0,str(code/'src'))
from flux_action.training.data import WindowDataset
from flux_action.data.droid.index import ROWS_FILENAME
from flux_action.training.trainer import TrainConfig,Trainer
from flux_action.config import PolicyConfig
from flux_action.data.droid.episodes import CAPTION_SEPARATOR
records={}
for task,ids in [('cover_blocks',range(300,400)),('match_and_pick_from_conveyor',range(1600,1700))]:
    old=R/'assets/history-index-preflight-v1'/task
    m=json.loads((old/'manifest.json').read_text());m.pop('warning',None)
    assert [e['episode_index'] for e in m['episodes']]==list(ids)
    idx=R/'assets/official-history-index-v1'/task;idx.mkdir(parents=True,exist_ok=True)
    target=idx/ROWS_FILENAME
    if not target.exists():os.link(old/'rows.npy',target)
    assert sha(target)==sha(old/'rows.npy')
    m['rows']={'file':ROWS_FILENAME,'sha256':sha(target),'width':28}
    m['adapter_note']='Only v3 metadata mapped to upstream index. Complete history/future windows, no padding or custom masks; native WindowDataset sampling.'
    cfg=json.loads((R/f'configs/{task}-10k-prefix-v1.json').read_text())
    cfg.update(dataset='index',index_dir=str(idx),output_dir=str(S/'runs'/f'{task}-10k-official-history-v1'))
    assert not Path(cfg['output_dir']).exists(),'Never prepare over an existing official run'
    cfg['policy'].pop('prefix_memory_spec')
    mapping={'head':'observation.images.cam_high','left_wrist':'observation.images.cam_left_wrist','right_wrist':'observation.images.cam_right_wrist'}
    short=[]
    for ep in m['episodes']:
        src=ep['source_episode'];available=ep['n_frames'];ep['videos']={}
        ep['caption']=CAPTION_SEPARATOR.join(src['tasks'])
        for name,cam in mapping.items():
            k='videos/'+cam;first=round(src[k+'/from_timestamp']*25);last=round(src[k+'/to_timestamp']*25)
            assert abs(first/25-src[k+'/from_timestamp'])<1e-6
            available=min(available,last-first)
            if last-first<ep['n_frames']:short.append([ep['episode_index'],cam,last-first,ep['n_frames']])
            ep['videos'][name]={'file':f"{m['source_prefix']}/videos/{cam}/chunk-{src[k+'/chunk_index']:03d}/file-{src[k+'/file_index']:03d}.mp4",'first_frame':first}
        ep['valid_ranges']=[[1,available]]
    (idx/'manifest.json').write_text(json.dumps(m,indent=2))
    stats={'format_version':1,'normalization':'range','clip':6.0,'episodes':100,'frames':m['total_frames'],
           'action':{'parameterization':'absolute','absolute_dims':[],**cfg['policy']['action_normalization']},'state':cfg['policy']['state_normalization']}
    (idx/'statistics.json').write_text(json.dumps(stats,indent=2))
    pc=PolicyConfig(**cfg['policy']);pc.validate_training();assert pc.n_obs_steps==8 and pc.history_snapshots==2
    assert not hasattr(pc,'prefix_memory_spec')
    tc=TrainConfig(**cfg);assert tc.global_batch(2)==32
    # Native reconciliation validates action dimensions and normalization.
    dummy=object.__new__(Trainer);dummy.manifest=m;dummy._log_line=lambda r:None
    Trainer._reconcile_with_index(dummy,cfg['policy'],idx)
    ds=WindowDataset(m,cfg['source_root'],target,seed=42,n_obs_steps=8,decoder='pyav',frame_hw=(480,640),visits_per_epoch=64,num_workers=2,world_size=2,windows_per_rank=2,grad_accumulation=8)
    counts=[];decoded=[]
    for i,ep in enumerate(ds.episodes):
        starts=ds.valid_starts(i);assert starts and starts[0]==8
        assert starts[-1]+32<ep['valid_ranges'][0][1]
        counts.append(len(starts))
    # Decode actual complete head and tail windows; native row/action alignment.
    for i in [0,24,99]:
        for start in [ds.valid_starts(i)[0],ds.valid_starts(i)[-1]]:
            item=ds.window_at(i,start);ep=ds.episodes[i];rows=ds.rows[ep['from_index']:ep['from_index']+ep['n_frames']]
            assert np.array_equal(item['state'].numpy(),rows[start-7:start+1,:14])
            assert np.array_equal(item['command_history'].numpy(),rows[start-8:start,14:])
            assert np.array_equal(item['action'].numpy(),rows[start:start+32,14:])
            assert not any('prefix' in k or 'mask' in k for k in item)
            for cam in mapping:assert tuple(item['images.'+cam].shape)==(40,3,480,640)
            decoded.append({'episode':ep['episode_index'],'start':start})
    path=R/f'configs/{task}-10k-official-history-v1.json';path.write_text(json.dumps(cfg,indent=2))
    records[task]={'episodes':100,'eligible_starts':sum(counts),'short_camera_spans':short,'decoded_native_windows':decoded,'config':str(path),'config_sha256':sha(path),'rows_sha256':sha(target),'new_output':cfg['output_dir']}
files={str(p.relative_to(R)):sha(p) for p in (code/'src').rglob('*.py')}
for n in ['train_official_history.py','audit_official_history.py','check_official_history_gate.py','launch_train_official_history.sh','async_cpu_ema.py','cpu_power_ema.py']:
    files['scripts/'+n]=sha(R/'scripts'/n)
for task in records:
    for p in [R/f'configs/{task}-10k-official-history-v1.json',*(R/'assets/official-history-index-v1'/task).glob('*.json')]:files[str(p.relative_to(R))]=sha(p)
result={'time':time.time(),'ready':True,'official_commit':'830f1ce178d7aea52f18fde31d8e8e36ce6bae21','pristine_policy_loss_trainer_dataset':True,'custom_prefix':False,'custom_temporal_loss':False,'fresh_base_optimizer':True,'records':records,'files':files,'scope':'Official supported history mechanism with RoboDojo 14D/3camera/25Hz metadata and retained experiment budget. Not the SO101 robot-specific LoRA recipe. CPU EMA/offload and safe stop are operational wrappers, not new conditioning.'}
(R/'status/official-history-ready.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['records']),flush=True)
