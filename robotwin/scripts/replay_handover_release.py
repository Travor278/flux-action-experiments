"""One fixed recorded handover prefix, oracle vs model 16/32-step continuation."""
import gc,importlib.util,json,os,pathlib,sys,time,traceback,numpy as np,torch
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/handover-release-replay-v1';R.mkdir(exist_ok=True);A=B/'evaluation/policy-error-audit-v1'
os.chdir(B/'RoboTwin')
for d in [B,B/'RoboTwin',B/'XPolicyLab']:sys.path.insert(0,str(d))
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12,0)
spec=importlib.util.spec_from_file_location('release_replay_upstream',B/'RoboTwin/scripts/eval_policy_xpolicylab.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
args,_=u.load_task_args({'task_name':'handover_block','task_config':'demo_clean','policy_name':'ReleaseReplayDiagnostic','ckpt_setting':'handover-release-replay-v1'})
args.update(save_data=False,collect_data=False,render_freq=0,eval_mode=True,need_plan=True,save_path=str(R/'sim-cache'))
plan=json.loads((A/'plan.json').read_text());p=next(x for x in plan['probes'] if x['task']=='handover_block' and x['episode']==0 and x['phase']=='tail_16')
raw=np.load(R/'source-episode0.npz');inp=np.load(p['input']);pred=np.load(A/f'{p["id"]}-train-4-prediction.npz')['prediction'];seed=json.loads((R/'source.json').read_text())['seed'];start=p['frame'];assert np.array_equal(raw['state'][start],inp['state'])
keys=['left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state'];edges=[0,6,7,13,14]
def save(path,d):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(path)
def act(env,obs,cmd):
 a={k:cmd[edges[j]:edges[j+1]].copy() for j,k in enumerate(keys)};flat,kind=u.xpolicylab_action_to_robotwin(a,action_type='joint',current_observation=obs);assert kind=='qpos' and np.array_equal(flat.astype(np.float32),cmd.astype(np.float32));env.take_action(flat,action_type=kind);return env.get_obs()
results=[]
try:
 for variant in ['oracle','model4_16','model4_32']:
  file=R/f'{variant}.json'
  if file.exists():results.append(json.loads(file.read_text()));continue
  commands=inp['target'] if variant=='oracle' else pred[:16] if variant=='model4_16' else pred
  env=None;t0=time.time()
  try:
   env=u.class_decorator('handover_block');env.setup_demo(now_ep_num=0,seed=seed,is_test=True,**args);env.step_lim=800;obs=env.get_obs()
   for i,cmd in enumerate(raw['action'][:start]):
    obs=act(env,obs,cmd);assert not env.eval_success,'Prefix completed before diagnostic observation'
    if i%20==0:save(B/'status/handover-release-replay-progress.json',{'time':time.time(),'variant':variant,'phase':'expert_prefix','frame':i,'total':start})
   ref=np.asarray(inp['state']);q=np.asarray(obs['joint_action']['vector']);box=np.asarray(env.box.get_functional_point(0,'pose').p);target=np.asarray(env.target_box.get_functional_point(1,'pose').p)
   r={'variant':variant,'seed':seed,'context_frame':start,'prefix_state_max_error':float(np.max(np.abs(q-ref))),'prefix_box_minus_target':(box-target).tolist(),'prefix_head_rgb_mae':float(np.abs(obs['observation']['head_camera']['rgb'].astype(float)-inp['cam_head'].astype(float)).mean()),'trace':[],'horizon':len(commands),'teacher_forced_prefix':True}
   success=False
   for i,cmd in enumerate(commands):
    obs=act(env,obs,cmd);q=np.asarray(obs['joint_action']['vector']);box=np.asarray(env.box.get_functional_point(0,'pose').p);success=bool(env.eval_success or env.check_success());r['trace'].append({'offset':i,'box_minus_target':(box-target).tolist(),'right_gripper':float(q[13]),'success':success})
    if success:break
   r.update(complete=True,success=success,executed=len(r['trace']),wall_seconds=time.time()-t0);save(file,r);results.append(r)
   save(B/'status/handover-release-replay-progress.json',{'time':time.time(),'completed':len(results),'expected':3,'last':{k:v for k,v in r.items() if k!='trace'}})
  finally:
   if env is not None:
    try:env.close_env(clear_cache=True)
    except Exception:pass
    for n in ['robot','cameras','scene']:
     if hasattr(env,n):setattr(env,n,None)
    env=None;gc.collect();torch.cuda.empty_cache()
 save(B/'status/handover-release-replay-complete.json',{'time':time.time(),'complete':True,'scope':'One training-scene expert-prefix continuation, not autonomous success rate','results':results})
except Exception:save(B/'status/handover-release-replay-error.json',{'time':time.time(),'error':traceback.format_exc()});raise
