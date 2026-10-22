import hashlib,json,pathlib,time,shutil
from flux_action.training.trainer import TrainConfig
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
proof=json.loads((B/'status/two-gpu-feasibility-verified.json').read_text())
assert proof['verified'] and proof['save_resume_export'] and proof['world_size']==2
assert proof['global_batch']==32 and proof['param_dtype']=='float32' and proof['ema_device']=='cpu'
assert shutil.disk_usage(S).free>700*10**9, 'Need space for concurrent latest+in-progress checkpoints and milestone exports'
record={'time':time.time(),'verified':True,'initialization':'official base, fresh optimizer; independent of deleted 3k training states','tasks':{},'recipe':proof,'files':{}}
for task in ['click_bell','handover_block']:
 cfg=json.loads((B/f'configs/{task}.json').read_text())
 cfg.update(output_dir=str(S/f'runs/{task}-10k-2gpu'),steps=10000,cooldown_start=9000,cooldown_steps=1000,shard_size=2,windows_per_rank=1,grad_accumulation=16,checkpoint_every=1000,keep_checkpoints=1,log_every=10,resume='auto')
 assert not pathlib.Path(cfg['output_dir']).exists(), 'Do not prepare over an existing formal run'
 assert cfg['frozen_steps']==200 and cfg['trunk_warmup_steps']==600 and cfg['heads_warmup_steps']==200
 assert cfg['reseed_on_resume']==False and cfg['policy']['optimizer_lr']==1e-5
 target=B/f'configs/{task}-10k-2gpu.json';target.write_text(json.dumps(cfg,indent=2))
 for limit in [1000,3000,5000,10000]:
  parsed=TrainConfig.from_file(str(target),[f'steps={limit}'])
  assert parsed.global_batch(2)==32 and parsed.cooldown_start==9000
 info=json.loads((B/f'assets/lerobot/{task}/meta/info.json').read_text());assert info['total_episodes']==50
 record['tasks'][task]={'episodes':50,'frames':info['total_frames'],'steps':10000,'global_batch':32,'sampled_windows':320000,'config':str(target),'output_dir':cfg['output_dir']}
for rel in ['scripts/train_two_gpu_worker.py','scripts/cpu_power_ema.py','scripts/launch_train_10k.sh','scripts/verify_10k_export.py','configs/click_bell-10k-2gpu.json','configs/handover_block-10k-2gpu.json']:
 record['files'][rel]=hashlib.sha256((B/rel).read_bytes()).hexdigest()
(B/'status/ten-k-ready.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

