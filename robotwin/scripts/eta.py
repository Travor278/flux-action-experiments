import json,pathlib,statistics,time
S=pathlib.Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
out={}
for task in ['click_bell','handover_block']:
 p=S/'runs'/task/'metrics.jsonl'
 if not p.exists(): out[task]={'phase':'not_started'};continue
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 rows=[r for r in rows if 'step_time' in r][-20:]
 if not rows: out[task]={'phase':'initializing'};continue
 speed=statistics.median((b['time']-a['time'])/(b['step']-a['step']) for a,b in zip(rows,rows[1:])) if len(rows)>1 else rows[0]['step_time']
 last=rows[-1]
 out[task]={'phase':'exported' if (p.parent/'TRAIN_COMPLETE').exists() else 'training_or_saving',
            'step':last['step'],'loss':last['loss'],'grad_norm':last['grad_norm'],
            'last_update_age_seconds':time.time()-last['time'],'seconds_per_update':speed,
            'eta_seconds':max(0,3000-last['step'])*speed}
(B/'status/eta.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
