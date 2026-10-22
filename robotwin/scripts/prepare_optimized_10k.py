import pathlib,json,hashlib,time,shutil
from flux_action.training.trainer import TrainConfig
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
proof=json.loads((B/'status/efficiency-selected-verified.json').read_text())
assert all(proof[k] for k in ['complete','save_resume_export','safe_stop_verified'])
best=proof['selected'];assert best['valid'] and best['micro'] in [1,2,4] and best['ema_mode'] in ['buffered','async']
assert best['seconds_per_update']<20 and set(best['gpu_util_mean'])=={'0','1'},'Review weak performance before long training'
assert shutil.disk_usage(S).free>700*10**9
record={'time':time.time(),'verified':True,'selected':best,'proof':str(B/'status/efficiency-selected-verified.json'),'initialization':'official base, fresh optimizer; old stopped 660-step runs have no checkpoint','files':{},'tasks':{}}
for task in ['click_bell','handover_block']:
 cfg=json.loads((B/f'configs/{task}-10k-2gpu.json').read_text())
 cfg.update(output_dir=str(S/f'runs/{task}-10k-opt-v1'),windows_per_rank=best['micro'],grad_accumulation=32//(2*best['micro']),prefetch_factor=best['prefetch'],resume='auto')
 assert not pathlib.Path(cfg['output_dir']).exists(), 'Never overwrite current training'
 p=B/f'configs/{task}-10k-opt-v1.json';p.write_text(json.dumps(cfg,indent=2))
 for limit in [100,1000,3000,5000,10000]:
  c=TrainConfig.from_file(str(p),[f'steps={limit}']);assert c.global_batch(2)==32 and c.cooldown_start==9000
 record['tasks'][task]={'config':str(p),'run':cfg['output_dir'],'steps':10000,'global_batch':32,'early_checkpoint':100}
 record['files'][str(p.relative_to(B))]=hashlib.sha256(p.read_bytes()).hexdigest()
for rel in ['scripts/train_efficiency_worker.py','scripts/cpu_power_ema.py','scripts/buffered_cpu_ema.py','scripts/async_cpu_ema.py','scripts/launch_train_optimized.sh','scripts/verify_optimized_export.py']:
 record['files'][rel]=hashlib.sha256((B/rel).read_bytes()).hexdigest()
(B/'status/optimized-ten-k-ready.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
