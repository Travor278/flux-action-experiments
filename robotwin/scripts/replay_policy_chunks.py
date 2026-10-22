"""Oracle-vs-prediction continuation from fixed expert prefixes. No model loading."""
import gc,importlib.util,json,os,pathlib,sys,time,traceback
import numpy as np
import torch
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
R=B/'evaluation/policy-chunk-replay-v1';R.mkdir(exist_ok=True);A=B/'evaluation/policy-error-audit-v1';E=B/'evaluation/expert-replay-click'
os.chdir(B/'RoboTwin')
for p in [B,B/'RoboTwin',B/'XPolicyLab']:sys.path.insert(0,str(p))
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12,0)
spec=importlib.util.spec_from_file_location('chunk_replay_upstream',B/'RoboTwin/scripts/eval_policy_xpolicylab.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
plan=json.loads((A/'plan.json').read_text());source=json.loads((E/'plan.json').read_text());args,_=u.load_task_args({'task_name':'click_bell','task_config':'demo_clean','policy_name':'OfflineChunkDiagnostic','ckpt_setting':'policy-chunk-replay-v1'})
args.update(save_data=False,collect_data=False,render_freq=0,eval_mode=True,need_plan=True,save_path=str(R/'sim-cache'))
keys=['left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state'];edges=[0,6,7,13,14]
records=[]
def save(path,x):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(path)
def act(env,obs,command):
 a={k:command[edges[j]:edges[j+1]].copy() for j,k in enumerate(keys)}
 flat,kind=u.xpolicylab_action_to_robotwin(a,action_type='joint',current_observation=obs)
 assert np.array_equal(flat.astype(np.float32),command.astype(np.float32)) and kind=='qpos'
 env.take_action(flat,action_type=kind);return env.get_obs()
try:
 for ep in [0,1,2,3]:
  p=next(x for x in plan['probes'] if x['task']=='click_bell' and x['episode']==ep and x['phase']=='last_valid');assert p['valid_training_start']
  src=next(x for x in source['episodes'] if x['episode']==ep);raw=np.load(src['npz']);inp=np.load(p['input']);start=p['frame'];assert np.array_equal(raw['state'][start],inp['state'])
  for variant in ['oracle','model4','model16']:
   path=R/f'episode-{ep}-{variant}.json'
   if path.exists():records.append(json.loads(path.read_text()));continue
   if variant=='oracle':commands=inp['target']
   else:
    steps=int(variant[5:]);pred=A/f'{p["id"]}-train-{steps}-prediction.npz';assert pred.exists();commands=np.load(pred)['prediction'][:len(inp['target'])]
   assert commands.ndim==2 and commands.shape[1]==14 and np.isfinite(commands).all()
   env=None;t0=time.time()
   try:
    env=u.class_decorator('click_bell');env.setup_demo(now_ep_num=ep,seed=src['seed'],is_test=True,**args);env.step_lim=400;obs=env.get_obs()
    for command in raw['action'][:start]:obs=act(env,obs,command);assert not env.eval_success,'Expert prefix already finished task'
    before=np.asarray(obs['joint_action']['vector'],dtype=np.float32);state_error=float(np.max(np.abs(before-inp['state'])))
    result={'episode':ep,'seed':src['seed'],'variant':variant,'context_frame':start,'prefix_state_max_error':state_error,'prediction_horizon':len(commands),'source_observation_in_train':True,'trace':[],'oracle_conditioned_prefix':True}
    success=False;arm=src['source_scene']['info']['{a}'];bell=np.asarray(env.bell.get_contact_point(0)[:3])
    for i,command in enumerate(commands):
     obs=act(env,obs,command);tcp=np.asarray(getattr(env.robot,f'get_{arm}_tcp_pose')()[:3]);q=np.asarray(obs['joint_action']['vector'])
     success=bool(env.eval_success or env.check_success());result['trace'].append({'offset':i,'success':success,'tcp_minus_bell':(tcp-bell).tolist(),'gripper':float(q[6 if arm=='left' else 13])})
     if success:break
    result.update(complete=True,success=success,executed=len(result['trace']),success_offset=result['trace'][-1]['offset'] if success else None,wall_seconds=time.time()-t0)
    save(path,result);records.append(result)
    save(B/'status/policy-chunk-replay-progress.json',{'time':time.time(),'completed':len(records),'expected':12,'last':{k:v for k,v in result.items() if k!='trace'}})
   finally:
    if env is not None:
     try:env.close_env(clear_cache=True)
     except Exception:pass
     for name in ['robot','cameras','scene']:
      if hasattr(env,name):setattr(env,name,None)
     env=None;gc.collect();torch.cuda.empty_cache()
 save(B/'status/policy-chunk-replay-complete.json',{'time':time.time(),'complete':True,'scope':'Teacher-forced training-prefix continuation, not autonomous test success','episodes':[0,1,2,3],'groups':{v:{'successes':sum(r['success'] for r in records if r['variant']==v),'episodes':4} for v in ['oracle','model4','model16']},'records':[{k:v for k,v in r.items() if k!='trace'} for r in records]})
except Exception:
 save(B/'status/policy-chunk-replay-error.json',{'time':time.time(),'error':traceback.format_exc()});raise
