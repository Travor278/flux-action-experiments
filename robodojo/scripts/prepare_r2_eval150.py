from pathlib import Path
import json,hashlib,time,subprocess,os,ast
from eval150_contract import TASKS,SEEDS,POLICY,TAG,official_config,output
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert os.environ.get('EVAL_NUM') in (None,'50','native')
dest=R/'XPolicyLab/policy'/POLICY;dest.mkdir(exist_ok=True)
for name,content in {'deploy.py':(R/'scripts/deploy_r2_eval150.py').read_bytes(),'deploy.yml':b'eval_batch: false\nprotocol: ws\n'}.items():
 p=dest/name
 if p.exists():assert p.read_bytes()==content
 else:p.write_bytes(content)
source_names=['eval150_contract.py','serve_r2_eval150.py','run_r2_eval150.py','check_r2_eval150_model.py','audit_r2_eval150.py','aggregate_r2_eval150.py','launch_r2_eval150.sh','launch_r2_eval150_pair.sh']
source={}
for name in source_names:
 p=R/'scripts'/name
 if p.suffix=='.py':ast.parse(p.read_text())
 source[str(p.relative_to(R))]=sha(p)
for name in ['deploy.py','deploy.yml']:source[str((dest/name).relative_to(R))]=sha(dest/name)
runtime={};repo=R/'RoboDojo'
for directory in ['env','env_cfg','utils','src/eval_client','task/RoboDojo']:
 for p in (repo/directory).rglob('*'):
  if p.is_file() and p.suffix in ('.py','.yml','.yaml','.json'):
   runtime[str(p.relative_to(R))]=sha(p)
for directory in ['flux-action-official-v1/src','XPolicyLab/client_server','XPolicyLab/utils','XPolicyLab/policy/FLUX_RoboDojo_History']:
 for p in (R/directory).rglob('*.py'):runtime[str(p.relative_to(R))]=sha(p)
# Record exact official source correspondence; do not silently accept modified task logic.
diff=subprocess.check_output(['git','-C',str(repo),'diff','--name-only','HEAD','--','env','env_cfg','utils','src/eval_client','task/RoboDojo'],text=True)
assert not diff.strip(),diff
tasks={}
os.environ['ROBODOJO_RUN_ID']='frozen-eval150-contract'
for task in TASKS:
 seeds={}
 for seed in SEEDS:
  assert not (output(R,task,seed)/'result.json').exists(),'Cannot replace plan after evaluating'
  root=repo/f'Assets/Eval_Layout/RoboDojo/arx_x5/{seed}'
  paths=sorted(root.glob(task+'_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]));assert len(paths)>=50
  cfg,effective=official_config(R,task,seed,12345)
  seeds[str(seed)]={'layout_ids':list(range(50)),'layout_files':{i:str(p) for i,p in enumerate(paths[:50])},'layout_sha256':{i:sha(p) for i,p in enumerate(paths[:50])},'available_layouts':len(paths),'effective_config_sha256':effective['core_sha256'],'effective_config':effective['config']}
 tasks[task]={'training_step':10000,'eval_num_per_seed':50,'seeds':seeds}
plan={'version':2,'tasks':tasks,'source_sha256':source,'runtime_sha256':runtime,'official_main_sha256':sha(repo/'src/eval_client/main.py'),'official_repo_revision':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'evaluation_design':'one frozen 10k EMA checkpoint per task, three official layout seeds0/1/2 each50; inference_seed retained fromexport, no trainingseed variance claim','metrics':['success_rate','Score=mean official per-episode score*100'],'aggregation':'mean/population std(ddof0) across3runs; sample std also recorded','error_policy':'Stop on unstable/unaccounted layout; preserve partial prefix and firstcause; no silent replacement or reroll','model_selection':'final10000 ema_0p10 only, chosen before test','control':'Native official config-assembly AST block, create_eval_env/reset/run_eval/close; read-only provenance/strict completeness wrapper','remaining_runtime_gate':'Actual final model load, new tasks complete-loop configuration and150case audits pending. No claim of published baseline calibration.'}
p=R/'status/r2-eval150-plan.json'
if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(plan))
else:p.write_text(json.dumps(plan,indent=2))
(R/'status/r2-eval150-prepared.json').write_text(json.dumps({'time':time.time(),'prepared':True,'plan_sha256':sha(p),'source_files':len(source),'runtime_files':len(runtime),'config_assemblies_verified':6,'episodes_planned':300,'runtime_executed':False},indent=2));print('EVAL150_PROTOCOL_PREPARED_NOT_EVALUATED',flush=True)
