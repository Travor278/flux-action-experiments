import json,pathlib
B=pathlib.Path(__file__).resolve().parents[1]
S=pathlib.Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
S.mkdir(parents=True,exist_ok=True)
for task in ['click_bell','handover_block']:
    cfg={'source_root':str(B/'assets/lerobot'/task),'index_dir':str(B/'assets/index'/task),
         'output_dir':str(S/'runs'/task),'steps':3000,'cooldown_start':2500,'cooldown_steps':500,
         'frozen_steps':200,'trunk_warmup_steps':600,'heads_warmup_steps':200,
         'seed':42,'reseed_on_resume':False,'windows_per_rank':1,'grad_accumulation':8,
         'num_workers':2,'visits_per_epoch':64,'decoder':'pyav','frame_hw':[240,320],
         'shard_size':4,'param_dtype':'float32','compute_dtype':'bfloat16','reduce_dtype':'float32',
         'activation_checkpointing':True,'reshard_after_forward':True,'fused_optimizer':True,
         'max_grad_norm':1.0,'ema_sigma_rels':[0.1,0.05],'checkpoint_every':1000,
         'keep_checkpoints':1,'keep_every':None,'log_every':10,'resume':'auto',
         'policy':{'trunk_weights':str(B/'assets/flux-base/flux-3-action-base.safetensors'),
             'video_vae_id':str(B/'assets/flux-base/video_vae.safetensors'),
             'text_encoder_id':str(B/'assets/flux-base/text_encoder'),
             'action_dim':14,'action_modality':'action_robotwin','action_parameterization':'absolute',
             'gripper_flip_dims':[],'camera_layout':'grid',
             'camera_keys':['images.head','images.left_wrist','images.right_wrist'],
             'canvas_hw':[480,640],'chunk_size':32,'n_action_steps':16,'fps':15.0,
             'attn_mode':'torch','compile_model':False,'vae_batch_windows':1,
             'optimizer_lr':1e-5,'optimizer_lr_heads_multiplier':5.0,
             'augment':False,'sampler':'euler','num_inference_steps':4,
             'sampler_shift':5.0,'guidance_scale':3.0,'guidance_scale_action':1.0}}
    (B/'configs'/f'{task}.json').write_text(json.dumps(cfg,indent=2))
protocol={'benchmark':'RoboTwin 2.0','embodiment':'aloha-agilex','dataset':'demo_clean',
 'tasks':{'click_bell':'single active arm per episode; dual-arm scene','handover_block':'bimanual handover'},
 'train_episodes_per_task':50,'train_seed':42,'train_steps':3000,'global_batch':32,
 'eval_episodes_per_task':100,'eval_seed':0,'eval_seed_start':100000,'eval_instruction':'seen',
 'eval_max_steps':{'click_bell':400,'handover_block':800},'eval_workers':1,
 'eval_seed_rule':'upstream expert-filtered valid scenes; save and compare accepted seed lists across policies',
 'checkpoint_selection':'final step 3000, ema_0p10; no test-set selection',
 'comparison':'rerun pi05 on identical source revision, episodes, cameras, action convention and accepted eval seeds',
 'limitation':'Aloha-AgileX results are not PiperX results; published pi05 scores are contextual until full protocol matches',
 'flux_git':'830f1ce178d7aea52f18fde31d8e8e36ce6bae21',
 'robotwin_git':'ea8b21121ebb3cd201ff5b3fe361944ac94eda3f',
 'xpolicylab_git':'53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2'}
(B/'configs/protocol.json').write_text(json.dumps(protocol,indent=2))
