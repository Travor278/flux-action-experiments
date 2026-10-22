"""Postprocess saved predictions, preserving the distinction between pre/post success."""
import pathlib,json,time,numpy as np,hashlib
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/policy-error-audit-v1'
done=json.loads((B/'status/policy-error-audit-complete.json').read_text());assert done['complete'] and done['predictions']==480
plan=json.loads((R/'plan.json').read_text());assert hashlib.sha256((R/'plan.json').read_bytes()).hexdigest()==done['plan_sha256']
metrics=[json.loads(x) for x in (R/'metrics.jsonl').read_text().splitlines()];assert len(metrics)==480 and len({(r['id'],r['caption_variant'],r['inference_steps']) for r in metrics})==480
contact={r['episode']:r['first_success_frame'] for r in json.loads((B/'status/expert-replay-verified.json').read_text())['results']}
rows=[]
for p in plan['probes']:
 raw=np.load(p['input']);target=raw['target'];state=raw['state'];active=p['active_joint_indices'];grips=p['gripper_indices']
 for variant in ['train','alternate']:
  for steps in [4,16]:
   pred=np.load(R/f'{p["id"]}-{variant}-{steps}-prediction.npz')['prediction'];assert pred.shape==(32,14) and np.isfinite(pred).all()
   horizon=min(16,len(target));pre=True
   if p['task']=='click_bell':horizon=min(horizon,max(0,contact[p['episode']]-p['frame']+1));pre=horizon>0
   row={k:p[k] for k in ['id','task','episode','phase','frame','valid_training_start']};row.update(caption_variant=variant,steps=steps,pre_success=pre,horizon=horizon)
   if horizon:
    row['joint_mae']=float(np.abs(pred[:horizon,active]-target[:horizon,active]).mean());row['hold_joint_mae']=float(np.abs(state[active]-target[:horizon,active]).mean());row['gripper_mae']=float(np.abs(pred[:horizon,grips]-target[:horizon,grips]).mean())
   if p['task']=='click_bell':
    k=contact[p['episode']]-p['frame']
    if 0<=k<len(target):row['contact_command']={'offset':k,'inside_first16':k<16,'joint_mae':float(np.abs(pred[k,active]-target[k,active]).mean()),'pred_gripper':float(pred[k,grips[0]]),'target_gripper':float(target[k,grips[0]]),'pred_closed':bool(pred[k,grips[0]]<.2)}
   if p['task']=='handover_block':
    mask=target[:horizon,13]>.8;row['right_should_open']=int(mask.sum());row['right_open_correct']=int(((pred[:horizon,13]>.8)&mask).sum())
   rows.append(row)
groups=[]
for task,phase in sorted({(r['task'],r['phase']) for r in rows}):
 for variant in ['train','alternate']:
  for steps in [4,16]:
   rs=[r for r in rows if r['task']==task and r['phase']==phase and r['caption_variant']==variant and r['steps']==steps];good=[r for r in rs if r['pre_success']]
   group={'task':task,'phase':phase,'caption':variant,'steps':steps,'n':len(good),'excluded_after_expert_success':len(rs)-len(good)}
   for key in ['joint_mae','hold_joint_mae','gripper_mae']:group[key]=float(np.mean([r[key] for r in good])) if good else None
   cs=[r['contact_command'] for r in rs if 'contact_command' in r];group['contact_contexts']=len(cs)
   if cs:group.update(contact_joint_mae=float(np.mean([x['joint_mae'] for x in cs])),contact_closed_fraction=float(np.mean([x['pred_closed'] for x in cs])),contact_in_first16=sum(x['inside_first16'] for x in cs))
   if task=='handover_block':
    n=sum(r['right_should_open'] for r in rs);group['right_open_expected']=n;group['right_open_recall']=sum(r['right_open_correct'] for r in rs)/n if n else None
   groups.append(group)
report={'time':time.time(),'complete':True,'predictions_verified':480,'plan_sha256':done['plan_sha256'],'groups':groups,
 'caveats':['Training demonstration observations, not held-out rollout accuracy','Click metrics truncate at replay expert first-success command; post-success contexts excluded','Handover has no replay-derived success-time cutoff; phase interpretation required','One fixed inference noise seed; small per-phase sample of 10 episodes']}
(B/'status/policy-error-analysis.json').write_text(json.dumps(report,indent=2));(R/'analyzed-rows.json').write_text(json.dumps(rows,indent=2))
