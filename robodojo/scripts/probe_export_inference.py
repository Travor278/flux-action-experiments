"""Reload real EMA export and exercise offline + queued/reset RGB inference."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,time
import torch
from flux_action.policy import FluxActionPolicy
from flux_action.data.droid.video import decode_window
from robodojo_history_dataset import build_dataset

R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
root=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1'
out=Path(os.environ['FLUX_EXPORT_PROBE_DIR'])
assert json.loads((R/'status/full-preflight-complete.json').read_text())['complete']
torch.set_num_threads(4)
assert torch.cuda.device_count()==1
def phase(name,**kw):
    with (out/'phases.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'phase':name,**kw})+'\n')
phase('load_export')
policy=FluxActionPolicy.from_pretrained(root/'export-4-ema',device='cpu')
policy.set_text_encoder_offload(True)
policy.to('cuda').eval()
cfg=policy.config
assert cfg.prefix_memory_spec and cfg.action_dim==14 and cfg.fps==25
phase('export_loaded',dtype=cfg.torch_dtype,prefix=cfg.prefix_memory_spec)
results=[];datasets={}
for task in ('cover_blocks','match_and_pick_from_conveyor'):
    dc=SimpleNamespace(index_dir=str(R/'assets/history-index-preflight-v1'/task),source_root=str(R/'official-source'),policy=cfg.to_dict(),decoder='pyav',frame_hw=(480,640),visits_per_epoch=2)
    ds=build_dataset(dc,seed=42,rank=0,world_size=1,num_workers=1,windows_per_rank=1,grad_accumulation=1);datasets[task]=ds
    item=ds.window_at(0,min(100,ds.episodes[0]['n_frames']-1))
    batch={'task':[item['task']],'episode_tick':torch.tensor([item['episode_tick']],device='cuda')}
    for k in ['state','command_history','prefix_tick_indices']:
        batch[k]=item[k][None].to('cuda')
    for k in cfg.camera_keys:
        batch[k]=item[k][None,:cfg.n_obs_steps].to('cuda')
        batch['prefix.'+k]=item['prefix.'+k][None].to('cuda')
    began=time.time();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():a=policy.predict_action_chunk(batch)
    torch.cuda.synchronize();assert a.shape==(1,32,14) and torch.isfinite(a).all()
    result={'task':task,'shape':list(a.shape),'seconds':time.time()-began,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'action_sha256':hashlib.sha256(a.float().cpu().numpy().tobytes()).hexdigest()}
    results.append(result);phase('offline_inference_pass',**result)
    del item,batch,a
policy.reset()
ds=datasets['cover_blocks'];ep=ds.episodes[0];source=ep['source_episode'];frames={}
for cam,key in [('cam_high','images.head'),('cam_left_wrist','images.left_wrist'),('cam_right_wrist','images.right_wrist')]:
    k='videos/observation.images.'+cam
    path=R/'official-source'/f"{ds.manifest['source_prefix']}/{k}/chunk-{source[k+'/chunk_index']:03d}/file-{source[k+'/file_index']:03d}.mp4"
    frames[key]=decode_window(path,round(source[k+'/from_timestamp']*25),53,fps=25,frame_hw=(480,640),decoder='pyav').transpose(0,3,1,2).copy()
for tick in range(53):
    obs={k:torch.from_numpy(v[tick:tick+1]).to('cuda') for k,v in frames.items()}
    obs.update(state=torch.from_numpy(ds.rows[ep['from_index']+tick,:14].copy())[None].to('cuda'),task=[source['tasks'][0]])
    with torch.inference_mode():action=policy.select_action(obs)
    assert action.shape==(1,14) and torch.isfinite(action).all()
assert policy._prefix_memory.tick==52 and sorted(policy._prefix_memory.anchors)==[0,25,50]
policy.reset()
assert policy._prefix_memory.tick==-1 and not policy._prefix_memory.anchors and not policy._action_queue
item=datasets['match_and_pick_from_conveyor'].window_at(0,0)
obs={k:item[k][0:1].to('cuda') for k in cfg.camera_keys}
obs.update(state=item['state'][-1:,:].to('cuda'),task=[item['task']])
with torch.inference_mode():action=policy.select_action(obs)
assert torch.isfinite(action).all() and policy._prefix_memory.tick==0
phase('queue_and_episode_reset_pass',ticks=53)
report={'time':time.time(),'verified':True,'export':str(root/'export-4-ema'),'offline_cases':results,'queued_ticks':53,'reset_verified':True,'scope':'Functional real export inference with recorded observations, not closed-loop task success or a trained benchmark model'}
(out/'export-inference-verified.json').write_text(json.dumps(report,indent=2))
print('REAL_EXPORT_INFERENCE_QUEUE_RESET_PASS',flush=True)
