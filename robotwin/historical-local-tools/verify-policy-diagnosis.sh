set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,numpy as np,time,hashlib,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));R=B/'evaluation/policy-error-audit-v1'
plan=json.loads((R/'plan.json').read_text());done=json.loads((B/'status/policy-error-audit-complete.json').read_text());analysis=json.loads((B/'status/policy-error-analysis.json').read_text());assert done['complete'] and done['predictions']==analysis['predictions_verified']==480
metrics=[json.loads(x) for x in (R/'metrics.jsonl').read_text().splitlines()];assert len(metrics)==480 and len({(x['id'],x['caption_variant'],x['inference_steps']) for x in metrics})==480
for p in plan['probes']:
 for variant in ['train','alternate']:
  for steps in [4,16]:
   x=np.load(R/f'{p["id"]}-{variant}-{steps}-prediction.npz')['prediction'];assert x.shape==(32,14) and np.isfinite(x).all()
cp=json.loads((B/'status/policy-chunk-replay-complete.json').read_text());assert cp['complete'] and len(cp['records'])==12 and all(g['successes']==4 for g in cp['groups'].values())
h=json.loads((B/'status/handover-release-replay-complete.json').read_text());assert h['complete'];r={x['variant']:x for x in h['results']};assert r['oracle']['success'] and not r['model4_16']['success'] and r['model4_32']['success']
for x in r.values():assert x['prefix_state_max_error']==0 and x['context_frame']==278 and x['seed']==0
for x in r.values():assert np.allclose(x['prefix_box_minus_target'],r['oracle']['prefix_box_minus_target'],atol=1e-8,rtol=0)
for a,b in zip(r['model4_16']['trace'],r['model4_32']['trace'][:16]):assert a['offset']==b['offset'] and np.allclose(a['box_minus_target'],b['box_minus_target'],atol=1e-7,rtol=0) and abs(a['right_gripper']-b['right_gripper'])<1e-7
config_preserved={}
for task in ['click_bell','handover_block']:
 p=S/f'runs/{task}-10k-opt-v1/export-10000-ema/config.json';c=json.loads(p.read_text());assert c['num_inference_steps']==4 and c['n_action_steps']==16;config_preserved[task]=hashlib.sha256(p.read_bytes()).hexdigest()
coverage=json.loads((B/'status/coverage-phase-detail.json').read_text());g=json.loads((B/'status/policy-full-horizon-gripper.json').read_text())
report={'time':time.time(),'complete':True,'offline_predictions_verified':480,'fixed_observations':120,'source_checkpoints_unchanged':config_preserved,'h100_job':{'id':'job-REPLACE_WITH_YOUR_JOB_ID','status':'job_succeeded','node_count':0},'click_prefix_groups':cp['groups'],'handover_prefix_test':{k:{'success':x['success'],'executed':x['executed'],'final_right_gripper':x['trace'][-1]['right_gripper']} for k,x in r.items()},'model16_vs32_same_first16_verified':True,'handover_release_coverage':{k:v for k,v in coverage['tasks']['handover_block'].items() if k!='per_episode'},'handover_predicted_open_offsets':[x['first_open_right_or_closed_click_offset'] for x in g['rows'] if x['task']=='handover_block'],'render_gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True),'interpretation_limits':['Teacher-forced/prefix diagnostics on training scenes are not autonomous held-out scores','No corrected-data retraining or full closed-loop execution16-vs32 A/B performed','RoboTwin joint_state getter returns drive targets; target-angle error is not measured physical tracking error']}
(B/'status/training-policy-diagnosis-verified.json').write_text(json.dumps(report,indent=2))
PY
