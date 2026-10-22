set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time,numpy as np
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/expert-replay-click';v=json.loads((B/'status/expert-replay-verified.json').read_text());p=json.loads((R/'plan.json').read_text());rows=[]
for r,rec in zip(v['results'],p['episodes']):
 last_start=rec['frames']-33;rows.append({'episode':r['episode'],'last_train_observation_start':last_start,'first_replay_success_frame':r['first_success_frame'],'success_frame_outside_training_observation_starts':r['first_success_frame']>last_start})
v['training_tail_followup']=rows;assert all(x['success_frame_outside_training_observation_starts'] for x in rows);(B/'status/expert-replay-verified.json').write_text(json.dumps(v,indent=2))
old=B/'status/expert-replay-source-inspection.json';before=old.stat().st_size;d=json.loads(old.read_text())
for task,item in d['tasks'].items():item['aux_contents']={k:z for k,z in item.get('aux_contents',{}).items() if k.endswith('/seed.txt') or k.endswith('/scene_info.json')}
old.write_text(json.dumps(d,indent=2));(B/'status/expert-replay-inspection-cleanup.json').write_text(json.dumps({'time':time.time(),'path':str(old),'before_bytes':before,'after_bytes':old.stat().st_size,'reason':'Discard duplicated instruction payload strings; source archives and audit summaries preserved'}))
PY
