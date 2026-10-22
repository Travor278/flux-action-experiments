set -euo pipefail
python3 - <<'PY'
import pathlib,json,time,hashlib,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
out={'time':time.time(),'tasks':{},'disk_free_bytes':shutil.disk_usage(S).free}
for task in ['click_bell','handover_block']:
 E=B/f'evaluation/{task}/flux-opt-10000-validation20';r=json.loads((E/'result.json').read_text());p=json.loads((E/'paired_seeds.json').read_text());ref=json.loads((B/f'evaluation/{task}/flux3000-validation20/paired_seeds.json').read_text())
 assert r['complete'] and r['episodes']==len(p)==20 and len({x['seed'] for x in p})==20
 assert [(x['seed'],x['instruction']) for x in p]==[(x['seed'],x['instruction']) for x in ref]
 assert sum(x['success'] for x in p)==r['successes']
 out['tasks'][task]={'successes':r['successes'],'episodes':20,'paired_reference_verified':True,'result_sha256':hashlib.sha256((E/'result.json').read_bytes()).hexdigest(),'retained_partial_episodes':r.get('retained_partial_episodes',0)}
(B/'status/flux-10k-validation-paired-audit.json').write_text(json.dumps(out,indent=2))
PY
