"""Fixed observational action-error comparisons; no optimizer or simulator."""
import gc,hashlib,json,pathlib,time,numpy as np,torch
from flux_action.policy import FluxActionPolicy
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));R=B/'evaluation/policy-error-audit-v1'
plan=json.loads((R/'plan.json').read_text());ready=json.loads((B/'status/policy-error-audit-ready.json').read_text());assert ready['complete'] and ready['plan_sha256']==hashlib.sha256((R/'plan.json').read_bytes()).hexdigest()
torch.set_num_threads(4);started=time.time();records=[];out=R/'metrics.jsonl'
assert not out.exists(),'Do not repeat a partial diagnostic without explicit resume'
policy=None;loaded_task=None;checkpoints={}
def norm(x):return np.clip(2*(x-q01)/span-1,-cfg.normalization_clip,cfg.normalization_clip)
mapping={'images.head':'cam_head','images.left_wrist':'cam_left_wrist','images.right_wrist':'cam_right_wrist'}
for p in plan['probes']:
 if loaded_task!=p['task']:
  if policy is not None:del policy;gc.collect();torch.cuda.empty_cache()
  checkpoint=S/f'runs/{p["task"]}-10k-opt-v1/export-10000-ema'
  policy=FluxActionPolicy.from_pretrained(str(checkpoint),device='cuda');assert policy.config.torch_dtype=='bfloat16'
  cfg=policy.config;q01=np.array(cfg.action_normalization['q01']);span=np.array(cfg.action_normalization['q99'])-q01;span=np.where(span>1e-6,span,1.)
  loaded_task=p['task'];checkpoints[loaded_task]=str(checkpoint)
 assert hashlib.sha256(pathlib.Path(p['input']).read_bytes()).hexdigest()==p['input_sha256']
 raw=np.load(p['input']);target=raw['target'];state=raw['state'];h=min(16,len(target));target=target[:h];active=p['active_joint_indices'];g=p['gripper_indices'];inactive=[x for x in list(range(6))+list(range(7,13)) if x not in active]
 baseline=np.repeat(state[None],h,axis=0)
 for variant in plan['caption_variants']:
  batch={'state':torch.from_numpy(state[None]).to('cuda'),'task':[p['train_caption'] if variant=='train' else p['alternate_caption']]}
  for key,cam in mapping.items():batch[key]=torch.from_numpy(np.ascontiguousarray(raw[cam].transpose(2,0,1)[None])).to('cuda')
  for steps in plan['inference_steps']:
   cfg.num_inference_steps=steps;cfg.inference_seed=plan['inference_seed'];t=time.time()
   with torch.inference_mode():prediction=policy.predict_action_chunk(batch).float().cpu().numpy()[0]
   assert prediction.shape==(32,14) and np.isfinite(prediction).all()
   pred=prediction[:h];expert_closed=target[:,g]<.2
   ref_delta=(target[-1]-state)[active]/span[active];pred_delta=(pred[-1]-state)[active]/span[active];den=float(ref_delta@ref_delta)
   record={**{k:p[k] for k in ['id','task','episode','phase','frame','valid_training_start','side']},'caption_variant':variant,'inference_steps':steps,'horizon':h,'wall_seconds':time.time()-t,
    'active_arm_mae_rad':float(np.abs(pred[:,active]-target[:,active]).mean()),'inactive_arm_mae_rad':float(np.abs(pred[:,inactive]-target[:,inactive]).mean()) if inactive else None,
    'left_arm_mae_rad':float(np.abs(pred[:,:6]-target[:,:6]).mean()),'right_arm_mae_rad':float(np.abs(pred[:,7:13]-target[:,7:13]).mean()),
    'hold_state_active_arm_mae_rad':float(np.abs(baseline[:,active]-target[:,active]).mean()),'active_gripper_mae':float(np.abs(pred[:,g]-target[:,g]).mean()),
    'normalized_mse':float(np.mean((norm(pred)-norm(target))**2)),'hold_state_normalized_mse':float(np.mean((norm(baseline)-norm(target))**2)),
    'expert_closed_elements':int(expert_closed.sum()),'predicted_closed_when_expert_closed':int(((pred[:,g]<.2)&expert_closed).sum()),
    'first_action_gripper_prediction':pred[0,g].tolist(),'first_action_gripper_target':target[0,g].tolist(),
    'progress_projection_vs_expert':float(pred_delta@ref_delta/den) if den>1e-6 else None}
   np.savez_compressed(R/f'{p["id"]}-{variant}-{steps}-prediction.npz',prediction=prediction)
   with out.open('a') as f:f.write(json.dumps(record)+'\n')
   records.append(record);(B/'status/policy-error-audit-progress.json').write_text(json.dumps({'time':time.time(),'completed':len(records),'expected':plan['expected_predictions'],'last':record,'total_seconds':time.time()-started},indent=2));print(json.dumps(record),flush=True)
assert len(records)==plan['expected_predictions']
summary={'time':time.time(),'complete':True,'predictions':len(records),'checkpoints':checkpoints,'plan_sha256':ready['plan_sha256'],'elapsed_seconds':time.time()-started,'groups':[],'scope':plan['scope']}
for task,phase in sorted({(r['task'],r['phase']) for r in records}):
 for variant in plan['caption_variants']:
  for steps in plan['inference_steps']:
   rows=[r for r in records if r['task']==task and r['phase']==phase and r['caption_variant']==variant and r['inference_steps']==steps];closed=sum(r['expert_closed_elements'] for r in rows)
   metrics=['active_arm_mae_rad','inactive_arm_mae_rad','left_arm_mae_rad','right_arm_mae_rad','hold_state_active_arm_mae_rad','active_gripper_mae','normalized_mse','hold_state_normalized_mse','wall_seconds']
   summary['groups'].append({'task':task,'phase':phase,'caption_variant':variant,'inference_steps':steps,'n':len(rows),**{k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in metrics},'closed_recall':sum(r['predicted_closed_when_expert_closed'] for r in rows)/closed if closed else None,'expert_closed_elements':closed})
(B/'status/policy-error-audit-complete.json').write_text(json.dumps(summary,indent=2))
