import pathlib,json,hashlib,time,re,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
assert (B/'status/tail-repair-cpu-complete.txt').is_file()
d=json.loads((B/'status/tail-repair-data-verified.json').read_text());assert d['complete']
text=(B/'status/tail-repair-tests.log').read_text();m=re.search(r'(\d+) passed',text);assert m and int(m[1])>=33 and not re.search(r'\d+ failed|ERROR collecting',text)
r={'time':time.time(),'complete':True,'data_verified':True,'tests_passed':True,'test_summary':m.group(0),'files':{},'tasks':d['tasks'],
   'sampling':'equal episode visits, uniform ALL real ticks; repeat-last only for shape, real-target masks',
   'init':'official base, fresh optimizer; previous10k inference checkpoints untouched',
   'hyperparameters':'same2H100/micro4/accum4/global32/asyncFP32dualEMA/10k/lr1e-5+5e-5/seed42/chunk32execute16',
   'temporal_loss':'zero padded action/video losses with original fixed token-budget denominator; native14dim no dummy dimensions'}
paths=['scripts/robotwin_tail_dataset.py','scripts/launch_train_tail_repaired.sh','scripts/verify_tail_stage.py','scripts/train_tail_worker.py','scripts/async_cpu_ema.py','scripts/buffered_cpu_ema.py','scripts/cpu_power_ema.py']
paths += ['flux-action-tail-v1/src/flux_action/'+f for f in ['policy.py','processing/packing.py','training/trainer.py','training/checkpoint.py']]
for task in d['tasks']:
 paths += [f'configs/{task}-10k-tail-v1.json',f'assets/index-tail-v1/{task}/manifest.json',f'assets/index-tail-v1/{task}/rows.f32.npy',f'assets/index-tail-v1/{task}/statistics.json',f'assets/index-tail-v1/{task}/caption-provenance.json']
for p in paths:r['files'][p]=hashlib.sha256((B/p).read_bytes()).hexdigest()
assert shutil.disk_usage(str(B).replace('/hdd/','/ssd/')).free>700*10**9
(B/'status/tail-repair-ready.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
