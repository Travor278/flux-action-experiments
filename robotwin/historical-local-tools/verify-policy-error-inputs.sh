set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,hashlib,numpy as np,time,ast
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/policy-error-audit-v1';p=json.loads((R/'plan.json').read_text());assert len(p['probes'])==120 and p['expected_predictions']==480 and not (R/'metrics.jsonl').exists()
p['selected_source_episodes']={t:sorted({x['episode'] for x in p['probes'] if x['task']==t}) for t in ['click_bell','handover_block']}
for x in p['probes']:
 f=pathlib.Path(x['input']);assert hashlib.sha256(f.read_bytes()).hexdigest()==x['input_sha256'];z=np.load(f);assert z['state'].shape==(14,) and z['target'].shape[1]==14 and np.isfinite(z['state']).all() and np.isfinite(z['target']).all()
 for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:assert z[cam].shape==(240,320,3) and z[cam].dtype==np.uint8
 assert x['gripper_indices'] in [[6],[13],[6,13]]
(R/'plan.json').write_text(json.dumps(p,indent=2));r=json.loads((B/'status/policy-error-audit-ready.json').read_text());r['plan_sha256']=hashlib.sha256((R/'plan.json').read_bytes()).hexdigest();r['input_contract_verified']=True;r['selected_source_episodes']=p['selected_source_episodes'];r['script_sha256']={}
for name in ['prepare_policy_error_audit.py','run_policy_error_audit.py','launch_policy_error_audit.sh']:
 f=B/'scripts'/name
 if name.endswith('.py'):ast.parse(f.read_text())
 r['script_sha256'][name]=hashlib.sha256(f.read_bytes()).hexdigest()
r['verified_at']=time.time();(B/'status/policy-error-audit-ready.json').write_text(json.dumps(r,indent=2))
PY
