import gc,json,pathlib,time
import torch
from torch.utils.data._utils.collate import default_collate
from pi05_helpers import dataset_processors,verify_token_embedding

def main():
    B=pathlib.Path(__file__).resolve().parents[1]
    if not (B/'status/pi05-data-verified.json').exists():raise RuntimeError('Pi05 dataset validation not complete')
    from transformers.models.siglip import check
    assert check.check_whether_transformers_replace_is_installed_correctly()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy
    model_root=B.parent/'A2R_pi05_20261007/assets/pi05-base-b211f3d4'
    tokenizer=B.parent/'A2R_pi05_20261007/assets/paligemma-tokenizer'
    config=PreTrainedConfig.from_pretrained(model_root)
    assert isinstance(config,PI05Config)
    config.pretrained_path=str(model_root)
    config.input_features={};config.output_features={}
    config.dtype='bfloat16';config.device='cuda'
    config.freeze_vision_encoder=False;config.train_expert_only=False
    config.compile_model=False;config.chunk_size=32;config.n_action_steps=16
    results={};policy=None
    for task in ['click_bell','handover_block']:
        ds=LeRobotDataset(task,root=B/'assets/pi05-v3'/task,video_backend='pyav',download_videos=False,
                         delta_timestamps={'action':[i/15 for i in range(32)]})
        row=ds[0]
        assert tuple(row['observation.state'].shape)==(14,) and tuple(row['action'].shape)==(32,14)
        if policy is None:
            policy=make_policy(cfg=config,ds_meta=ds.meta)
            verify_token_embedding(policy,model_root)
            policy.eval()
        policy.reset()
        before,_=dataset_processors(config,ds.meta.stats,tokenizer)
        batch=before(default_collate([row]))
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.no_grad():loss,_=policy.forward(batch)
        torch.cuda.synchronize()
        assert torch.isfinite(loss).all()
        results[task]={'loss':float(loss),'seconds':time.perf_counter()-start,
                       'gpu':torch.cuda.get_device_name(),'peak_memory_gb':torch.cuda.max_memory_allocated()/2**30,
                       'state_dim':14,'action_dim':14,'chunk_size':32,'n_action_steps':16,
                       'stage':'base checkpoint forward only; no optimizer update or success-rate evaluation'}
        print(json.dumps(results[task]),flush=True)
    (B/'status/pi05-base-forward.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
