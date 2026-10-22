import hashlib,json,pathlib,time
import torch
from flux_action.models.video_vae import load_video_vae
from flux_action.processing.packing import materialize_video,encode_video,encode_single_frame,latent_hw
from flux_action.training.data import WindowDataset

def main():
    B=pathlib.Path(__file__).resolve().parents[1]
    weights=B/'assets/.flux-base.download/video_vae.safetensors'
    if not weights.exists():weights=B/'assets/flux-base/video_vae.safetensors'
    with weights.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest=='db1c3211548371cf42c7230d52b5a99df5072f246d6bdfdc8fee99208ef58f01'
    index=B/'assets/index/handover_block'
    manifest=json.loads((index/'manifest.json').read_text())
    ds=WindowDataset(manifest,B/'assets/lerobot/handover_block',index/'rows.f32.npy',seed=42,decoder='pyav',frame_hw=(240,320))
    w=ds.window(0)
    cameras=torch.stack([w['images.'+c] for c in ['head','left_wrist','right_wrist']])
    video=materialize_video(cameras,None,'cuda',layout='grid',canvas_hw=(480,640))
    vae=load_video_vae(str(weights),device='cuda',compile_model=False)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    latents=encode_video(vae,video,latent_hw('grid',(480,640)))
    condition=encode_single_frame(vae,video[:,0],latent_hw('grid',(480,640)),single_frame=True)
    torch.cuda.synchronize()
    assert tuple(latents.shape)==(1,96,9,15,20) and tuple(condition.shape)==(1,96,1,15,20)
    assert torch.isfinite(latents).all() and torch.isfinite(condition).all()
    record={'sha256':digest,'video_latents':list(latents.shape),'condition_latents':list(condition.shape),
            'gpu':torch.cuda.get_device_name(),'seconds':time.perf_counter()-started,
            'peak_memory_gb':torch.cuda.max_memory_allocated()/2**30,'finite':True,
            'scope':'real frozen VAE/NATTEN only; no full FLUX model or optimizer'}
    (B/'status/vae-preflight.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

if __name__=='__main__':main()
