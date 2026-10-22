set -euo pipefail
python3 - <<'PY'
import pathlib,json,statistics,math,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 rows=[json.loads(x) for x in (S/f'runs/{task}/metrics.jsonl').read_text().splitlines() if x.strip()]
 updates=[x for x in rows if 'step_time' in x]
 a={'points':{},'last':updates[-1],'all_finite':all(math.isfinite(x[k]) for x in updates for k in ['loss','grad_norm'])}
 for n in [100,500,1000,2000,3000]:
  part=[x for x in updates if n-100<x['step']<=n]
  a['points'][str(n)]={k:statistics.mean(x[k] for x in part) for k in ['loss','video_mse','action_mse','grad_norm']}
 tail=updates[-20:];a['steady_seconds_per_update']=statistics.median((y['time']-x['time'])/(y['step']-x['step']) for x,y in zip(tail,tail[1:]))
 out['tasks'][task]=a
(B/'status/three-k-loss-summary.json').write_text(json.dumps(out,indent=2))
PY
