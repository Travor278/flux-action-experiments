set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,numpy as np,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/policy-error-audit-v1';p=json.loads((R/'plan.json').read_text());rows=[]
for c in p['probes']:
 if c['phase']!='tail_16':continue
 x=np.load(R/f'{c["id"]}-train-4-prediction.npz')['prediction'];raw=np.load(c['input']);g=13 if c['task']=='handover_block' else c['gripper_indices'][0]
 event=x[:,g]>.8 if c['task']=='handover_block' else x[:,g]<.2;inds=np.flatnonzero(event)
 rows.append({'task':c['task'],'episode':c['episode'],'frame':c['frame'],'first_open_right_or_closed_click_offset':int(inds[0]) if len(inds) else None,'gripper_first':float(x[0,g]),'gripper_last16':float(x[15,g]),'gripper_last32':float(x[31,g]),'target_first':float(raw['target'][0,g]),'target_last':float(raw['target'][-1,g])})
(B/'status/policy-full-horizon-gripper.json').write_text(json.dumps({'time':time.time(),'rows':rows},indent=2))
PY
