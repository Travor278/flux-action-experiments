"""Matched RoboTwin pi05 full fine-tuning through the stock LeRobot trainer."""
import argparse,json,math,os,pathlib,sys,time
import torch
from pi05_helpers import dataset_processors,verify_token_embedding

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['click_bell','handover_block'])
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    B=pathlib.Path(__file__).resolve().parents[1]
    S=pathlib.Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
    sys.path.insert(0,str(B/'flux-action/src'))
    from matched_window_sampler import MatchedWindowSampler
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from transformers.models.siglip import check
    assert check.check_whether_transformers_replace_is_installed_correctly()
    data=B/'assets/pi05-v3'/args.task
    verified=json.loads((data/'PI05_DATA_VERIFIED.json').read_text())
    assert verified['episodes']==50 and verified['all_numeric_columns']=='exact match to FLUX source'
    manifest=json.loads((B/'assets/index'/args.task/'manifest.json').read_text())
    model_root=B.parent/'A2R_pi05_20261007/assets/pi05-base-b211f3d4'
    tokenizer=B.parent/'A2R_pi05_20261007/assets/paligemma-tokenizer'
    policy=PreTrainedConfig.from_pretrained(model_root)
    assert isinstance(policy,PI05Config)
    policy.pretrained_path=str(model_root);policy.input_features={};policy.output_features={}
    policy.dtype='bfloat16';policy.device='cuda';policy.push_to_hub=False;policy.use_peft=False
    policy.freeze_vision_encoder=False;policy.train_expert_only=False
    policy.gradient_checkpointing=True;policy.compile_model=False
    policy.chunk_size=32;policy.n_action_steps=16
    policy.scheduler_warmup_steps=100;policy.scheduler_decay_steps=args.steps
    run='pi05_'+args.task+'_seed42'
    cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id=args.task,root=str(data),video_backend='pyav'),
        policy=policy,output_dir=S/'runs'/run,job_name=run,batch_size=16,num_workers=4,
        steps=args.steps,eval_freq=0,log_freq=10,save_freq=1000,seed=42)
    cfg.validate();cfg.wandb.enable=False
    contract={'task':args.task,'episodes':50,'steps':args.steps,'global_batch':32,'gpus':2,'seed':42,
              'sampler':'identical FLUX episode visits/window RNG; same global batch multiset',
              'windows_per_epoch':3200,'chunk_size':32,'n_action_steps':16,'action_dim':14,
              'action_semantics':'native absolute joint positions and gripper fractions, no remapping',
              'normalization':'same exact global training q01/q99 as FLUX',
              'optimizer_lr':policy.optimizer_lr,'optimizer_betas':list(policy.optimizer_betas),
              'warmup_steps':policy.scheduler_warmup_steps,'output_dir':str(cfg.output_dir)}
    status=B/'status'/run;status.mkdir(parents=True,exist_ok=True)
    (status/'contract.json').write_text(json.dumps(contract,indent=2))
    print('PI05_MATCHED_CONFIG '+json.dumps(contract),flush=True)
    if args.preflight_only:return
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    assert world==2 and torch.cuda.device_count()==2,'Allocate exactly two H100 GPUs'
    torch.cuda.set_device(local)
    assert 'H100' in torch.cuda.get_device_name(local)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    original_adamw=torch.optim.AdamW
    class FusedAdamW(original_adamw):
        def __init__(self,*a,**kw):
            kw['fused']=True;kw['foreach']=False;super().__init__(*a,**kw)
    torch.optim.AdamW=FusedAdamW
    import lerobot.optim.optimizers as optimizers
    if hasattr(optimizers,'AdamW'):optimizers.AdamW=FusedAdamW
    original_loader=torch.utils.data.DataLoader
    class MatchedLoader(original_loader):
        def __init__(self,*a,**kw):
            dataset=a[0] if a else kw.get('dataset')
            if hasattr(dataset,'meta'):
                kw['sampler']=MatchedWindowSampler(manifest,seed=42,visits=64)
                kw['shuffle']=False;kw['drop_last']=True
                if kw.get('num_workers',0):
                    kw['persistent_workers']=True;kw['prefetch_factor']=2
            super().__init__(*a,**kw)
    torch.utils.data.DataLoader=MatchedLoader
    from lerobot.scripts import lerobot_train as trainer
    def processors(*a,**kw):
        config=kw.get('policy_cfg',a[0] if a else None)
        return dataset_processors(config,kw['dataset_stats'],tokenizer)
    trainer.make_pre_post_processors=processors
    original_policy=trainer.make_policy
    def verified_policy(*a,**kw):
        model=original_policy(*a,**kw);verify_token_embedding(model,model_root)
        total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable>=.98*total
        print('PI05_PARAMETERS '+json.dumps({'rank':rank,'total':total,'trainable':trainable}),flush=True)
        return model
    trainer.make_policy=verified_policy
    original_update=trainer.update_policy;step=0
    def update(*a,**kw):
        nonlocal step
        result=original_update(*a,**kw);step+=1
        metrics=result[0];values={}
        for key in ['loss','grad_norm']:
            metric=getattr(metrics,key);value=float(getattr(metric,'val',getattr(metric,'avg',metric)))
            assert math.isfinite(value),f'Nonfinite {key}';values[key]=value
        if step==1 or step%10==0:
            record={**values,'time':time.time(),'step':step,'rank':rank,'world_size':world,'global_batch':32,
                    'gpu_allocated_gb':torch.cuda.memory_allocated()/2**30}
            tmp=status/f'progress-rank{rank}.tmp';tmp.write_text(json.dumps(record));tmp.replace(status/f'progress-rank{rank}.json')
            print('PI05_UPDATE '+json.dumps(record),flush=True)
        return result
    trainer.update_policy=update
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    accelerator=Accelerator(mixed_precision='bf16',step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True,gradient_as_bucket_view=True)])
    trainer.train(cfg,accelerator)
    assert step==args.steps,'Trainer returned before the requested optimizer budget'
    if rank==0:
        (status/'COMPLETE.json').write_text(json.dumps({'steps':step,'output_dir':str(cfg.output_dir),'contract':contract},indent=2))

if __name__=='__main__':main()
