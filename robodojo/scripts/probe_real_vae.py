"""Standalone real-weight CUDA VAE prefix/tail-causality probe, not training."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,socket,subprocess,time
import torch
from flux_action.config import PolicyConfig
from flux_action.models.video_vae import load_video_vae
from flux_action.processing import history,packing,prefix_memory,temporal_masking
from robodojo_history_dataset import build_dataset

R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
B=R.parent/'flux_piperx_20261008'
out=Path(os.environ['FLUX_VAE_PROBE_DIR'])
def phase(name,**kw):
    with (out/'phases.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'phase':name,**kw})+'\n')
assert torch.cuda.device_count()==1
torch.set_num_threads(4)
weights=B/'assets/flux-base/video_vae.safetensors'
assert weights.is_file()
with weights.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
phase('loading_real_vae',weights=str(weights),sha256=digest)
vae=load_video_vae(str(weights),device='cuda',compile_model=False)
phase('real_vae_loaded',parameters=sum(p.numel() for p in vae.module.parameters()))
results=[]
cfg=PolicyConfig(camera_layout='grid',camera_keys=('images.head','images.left_wrist','images.right_wrist'),canvas_hw=(480,640),action_dim=14,action_modality='action_robodojo',gripper_flip_dims=(),inference_profile='history',n_obs_steps=8,history_snapshots=2,condition_on_past_actions=True,fps=25.,chunk_size=32,augment=False,prefix_memory_spec=prefix_memory.PrefixMemorySpec().to_dict(),vae_batch_windows=1)
for task,episode in [('cover_blocks',324),('match_and_pick_from_conveyor',1625)]:
    loadercfg=SimpleNamespace(index_dir=str(R/'assets/history-index-preflight-v1'/task),source_root=str(R/'official-source'),policy=cfg.to_dict(),decoder='pyav',frame_hw=(480,640),visits_per_epoch=2)
    ds=build_dataset(loadercfg,seed=42,rank=0,world_size=1,num_workers=1,windows_per_rank=1,grad_accumulation=1)
    pos=next(i for i,e in enumerate(ds.episodes) if e['episode_index']==episode)
    item=ds.window_at(pos,ds.episodes[pos]['n_frames']-6)
    batch={k:v[None] if isinstance(v,torch.Tensor) else torch.tensor([v]) if isinstance(v,int) else [v] for k,v in item.items()}
    cams=torch.stack([item[k] for k in cfg.camera_keys])
    video=packing.materialize_video(cams.float()/255,None,torch.device('cuda'),layout=cfg.camera_layout,canvas_hw=cfg.canvas_hw)[None]
    torch.cuda.reset_peak_memory_stats();began=time.time()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        latents=history.encode_windows(vae,video,cfg)
        am,vm=temporal_masking.prepare_masks(batch,[0],cfg,latents)
        prefix_tokens,prefix_ids=prefix_memory.encode_prefix(vae,batch,cfg,[0],torch.device('cuda'))
        # Alter ONLY synthetic future pixels. Every valid causal target latent
        # and separately encoded historical conditioning must remain identical.
        bad=~batch['video_valid_frames'][0,cfg.n_obs_steps:]
        changed=video.clone()
        indices=torch.nonzero(bad).flatten().to('cuda')+cfg.n_obs_steps
        changed[:,:,indices]=torch.randn_like(changed[:,:,indices]).clamp(-1,1)
        modified=history.encode_windows(vae,changed,cfg)
        torch.testing.assert_close(latents[:,:,:cfg.history_snapshots],modified[:,:,:cfg.history_snapshots],rtol=0,atol=0)
        valid=vm.reshape(1,-1,cfg.latent_hw[0]*cfg.latent_hw[1])[:,:,0]
        a=latents[:,:,cfg.history_snapshots:].permute(0,2,1,3,4)[valid]
        b=modified[:,:,cfg.history_snapshots:].permute(0,2,1,3,4)[valid]
        assert a.numel()>0
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        repeated,ids2=prefix_memory.encode_prefix(vae,batch,cfg,[0],torch.device('cuda'))
        torch.testing.assert_close(prefix_tokens,repeated,rtol=0,atol=0)
        assert torch.equal(prefix_ids,ids2) and prefix_ids[...,0].max()<0
        assert torch.isfinite(latents).all() and torch.isfinite(prefix_tokens).all()
    torch.cuda.synchronize()
    result={'task':task,'episode':episode,'start':item['start'],'seconds':time.time()-began,'valid_actions':int(am.sum()),'valid_future_latents':int(valid.sum()),'latents_shape':list(latents.shape),'prefix_shape':list(prefix_tokens.shape),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'causal_targets_exact':True,'prefix_repeat_exact':True}
    results.append(result);phase('case_pass',**result)
    del video,changed,latents,modified,prefix_tokens,repeated,item,batch,cams
    torch.cuda.empty_cache()
report={'time':time.time(),'verified':True,'hostname':socket.gethostname(),'weights_sha256':digest,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'cases':results,'diagnostic_geometry':cfg.to_dict(),'scope':'Real pretrained CUDA VAE and prefix/tail masks only; no DiT backward/optimizer/save-resume or expert-policy benchmark; geometry/H are diagnostic, not frozen'}
(out/'real-vae-verified.json').write_text(json.dumps(report,indent=2))
print('REAL_VAE_PREFIX_AND_CAUSAL_TAIL_PASS',flush=True)
