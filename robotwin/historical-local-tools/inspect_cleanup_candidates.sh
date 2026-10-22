set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
out={'time':time.time(),'candidates':[]}
for rel in ['assets/pi05-v21','runs/cpu-ema-smoke-cpu','runs/cpu-ema-smoke-cuda']:
 p=B/rel;files=[x for x in p.rglob('*') if x.is_file()]
 out['candidates'].append({'path':str(p),'files':len(files),'bytes':sum(x.stat().st_size for x in files),'symlink_files':sum(x.is_symlink() for x in files),'infos':{str(x.relative_to(p)):json.loads(x.read_text()) for x in p.glob('*/meta/info.json')}})
for task in ['click_bell','handover_block']:
 p=B/f'assets/pi05-v3/{task}/meta/info.json';out[task+'_v3_info']=json.loads(p.read_text())
out['proofs']=[str(x.name) for x in (B/'status').glob('*pi05*')]
(B/'status/cleanup-candidate-review.json').write_text(json.dumps(out,indent=2))
PY
