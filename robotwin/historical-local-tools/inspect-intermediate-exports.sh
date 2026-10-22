set -euo pipefail
python3 - <<'PY'
import pathlib,json,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'exports':[]}
for task in ['click_bell','handover_block']:
 for step in [1000,3000,5000,10000]:
  d=S/f'runs/{task}-10k-opt-v1/export-{step}-ema';files=[{'name':p.name,'bytes':p.stat().st_size,'links':p.stat().st_nlink} for p in d.iterdir() if p.is_file()];out['exports'].append({'task':task,'step':step,'path':str(d),'files':files,'manifest':json.loads((d/'manifest.json').read_text())})
(B/'status/flux-intermediate-export-cleanup-plan.json').write_text(json.dumps(out,indent=2))
PY
