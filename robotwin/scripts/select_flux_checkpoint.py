"""Select only after all paired validation cases complete; never inspect final test outcomes."""
import pathlib,json,time,hashlib
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
assert (B/'status/flux-milestone-validation-complete.txt').exists(),'Wait for all eight validations'
record={'time':time.time(),'selection_rule':'highest validation successes, ties resolved toward fewer training updates','validation_episodes':20,'final_test_episodes':100,'final_test_seed_parameter':0,'tasks':{}}
for task in ['click_bell','handover_block']:
 reference=json.loads((B/f'evaluation/{task}/flux3000-validation20/paired_seeds.json').read_text());results=[]
 for step in [1000,3000,5000,10000]:
  E=B/f'evaluation/{task}/flux-opt-{step}-validation20';r=json.loads((E/'result.json').read_text());pairs=json.loads((E/'paired_seeds.json').read_text());protocol=json.loads((E/'protocol.json').read_text())
  assert r['complete'] and r['episodes']==20 and r['split']=='validation' and r['policy']=='FLUX_Action'
  assert protocol['seed']==8 and protocol['test_num']==20 and protocol['instruction_type']=='seen' and protocol['task_config']=='demo_clean'
  assert [(p['seed'],p['instruction']) for p in pairs]==[(p['seed'],p['instruction']) for p in reference]
  assert sum(p['success'] for p in pairs)==r['successes']
  results.append({'step':step,'successes':r['successes'],'episodes':20,'validation_path':str(E)})
 best=min(results,key=lambda r:(-r['successes'],r['step']))
 checkpoint=S/f'runs/{task}-10k-opt-v1/export-{best["step"]}-ema'
 assert checkpoint.is_dir() and (checkpoint/'manifest.json').exists()
 record['tasks'][task]={'selected':best,'all_validation_results':results,'checkpoint':str(checkpoint),'final_test_output':str(B/f'evaluation/{task}/flux-selected-{best["step"]}-test100')}
target=B/'status/flux-checkpoint-selection.json'
if target.exists():
 old=json.loads(target.read_text());assert old['tasks']==record['tasks'],'Do not silently change a selection after final tests'
else:target.write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
