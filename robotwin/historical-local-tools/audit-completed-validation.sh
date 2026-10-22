set -euo pipefail
python3 - <<'PY'
import pathlib,json,time,hashlib,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'completed':[],'disk_free_bytes':shutil.disk_usage(S).free}
for step in [1000,3000,5000,10000]:
 for task in ['click_bell','handover_block']:
  E=B/f'evaluation/{task}/flux-opt-{step}-validation20';rp=E/'result.json'
  if not rp.exists():continue
  r=json.loads(rp.read_text())
  if not r.get('complete'):continue
  p=json.loads((E/'paired_seeds.json').read_text());ref=json.loads((B/f'evaluation/{task}/flux3000-validation20/paired_seeds.json').read_text());protocol=json.loads((E/'protocol.json').read_text())
  assert r['episodes']==len(p)==20 and r['split']=='validation' and len({x['seed'] for x in p})==20
  assert [(x['seed'],x['instruction']) for x in p]==[(x['seed'],x['instruction']) for x in ref]
  assert sum(x['success'] for x in p)==r['successes']
  assert protocol['seed']==8 and protocol['test_num']==20 and protocol['task_config']=='demo_clean' and protocol['instruction_type']=='seen'
  out['completed'].append({'task':task,'step':step,'successes':r['successes'],'episodes':20,'paired_reference_verified':True,'result_sha256':hashlib.sha256(rp.read_bytes()).hexdigest()})
out['top_level_transfer_candidates']=[]
for root in [B/'assets',S/'hf-publish',S/'hf-transfer-cache']:
 if root.exists():
  for x in root.glob('*'):
   if x.is_file() and x.name.endswith(('.partial','.part','.tmp','.tar')):out['top_level_transfer_candidates'].append(str(x))
(B/'status/flux-completed-validation-audit.json').write_text(json.dumps(out,indent=2))
PY
