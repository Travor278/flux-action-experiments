set -euo pipefail
python3 - <<'PY'
import pathlib,json,time,hashlib,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));task='click_bell';E=B/f'evaluation/{task}/flux-opt-5000-validation20'
r=json.loads((E/'result.json').read_text());p=json.loads((E/'paired_seeds.json').read_text());ref=json.loads((B/f'evaluation/{task}/flux3000-validation20/paired_seeds.json').read_text());protocol=json.loads((E/'protocol.json').read_text())
assert r['complete'] and r['episodes']==len(p)==20 and len({x['seed'] for x in p})==20
assert [(x['seed'],x['instruction']) for x in p]==[(x['seed'],x['instruction']) for x in ref]
assert sum(x['success'] for x in p)==r['successes']==3
assert protocol['seed']==8 and protocol['test_num']==20 and protocol['task_config']=='demo_clean'
candidates=[]
for root in [B/'assets',S/'hf-publish',S/'hf-transfer-cache']:
 if root.exists():
  for x in root.glob('*'):
   if x.is_file() and (x.name.endswith(('.partial','.part','.tmp','.tar'))):candidates.append({'path':str(x),'bytes':x.stat().st_size})
out={'time':time.time(),'task':task,'step':5000,'successes':3,'episodes':20,'paired_reference_verified':True,'result_sha256':hashlib.sha256((E/'result.json').read_bytes()).hexdigest(),'disk_free_bytes':shutil.disk_usage(S).free,'top_level_transfer_candidates':candidates,'deleted':[]}
(B/'status/flux-5k-click-validation-audit.json').write_text(json.dumps(out,indent=2))
PY
