set -euo pipefail
python3 - <<'PY'
import pathlib,json,time,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
candidates=[]
for pattern in ['*.tar','*.tar.gz','*.partial','*.incomplete']:
 for p in (B/'assets').glob(pattern):candidates.append({'path':str(p),'bytes':p.stat().st_size,'review_required':True})
out={'time':time.time(),'ssd_free_bytes':shutil.disk_usage(S).free,'transfer_candidates':candidates,'active_runs':{}}
for task in ['click_bell','handover_block']:
 r=S/f'runs/{task}-10k-opt-v1';out['active_runs'][task]={'checkpoints':[p.name for p in r.glob('step-*')],'exports':[p.name for p in r.glob('export-*')],'preserved':True}
(B/'status/active-file-audit.json').write_text(json.dumps(out,indent=2))
PY
