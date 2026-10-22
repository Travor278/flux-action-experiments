"""Training completion/model gate, immutable protocol, and explicit resume guard."""
from pathlib import Path
import hashlib,json,os,sys,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
task=sys.argv[1];assert task in ('cover_blocks','plug_in_charger')
plan=json.loads((R/'status/r2-eval150-plan.json').read_text())
eval_seed=int(os.environ['FLUX_EVAL_SEED']);assert eval_seed in (0,1,2)
for name,expected in plan['runtime_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected,name
for name,expected in plan['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected,name
train=S/'runs'/f'{task}-10k-official-history-r2-20261021';complete=json.loads((train/'TRAIN_COMPLETE.json').read_text())
assert complete['verified'] and complete['step']==10000 and complete['task']==task
export=train/'export-10000-ema';manifest=json.loads((export/'manifest.json').read_text())
for name,expected in manifest['sha256'].items():
 p=(export/name).resolve();assert p.is_relative_to(export.resolve())
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
cfg=json.loads((export/'config.json').read_text())
assert cfg['action_dim']==14 and cfg['n_obs_steps']==8 and cfg['n_action_steps']==16 and cfg['chunk_size']==32 and cfg['fps']==25 and cfg['num_inference_steps']==4
assert 'prefix_memory_spec' not in cfg and cfg['n_obs_steps']==8 and cfg['history_snapshots']==2
from eval150_contract import output,official_config
eval_seed=int(os.environ['FLUX_EVAL_SEED']);out=output(R,task,eval_seed);out.mkdir(parents=True,exist_ok=True)
protocol={'version':1,'task':task,'checkpoint':str(export),'checkpoint_sha256':manifest['sha256'],'training_step':10000,'layout_ids':plan['tasks'][task]['seeds'][str(eval_seed)]['layout_ids'],'layout_sha256':plan['tasks'][task]['seeds'][str(eval_seed)]['layout_sha256'],'seed':eval_seed,'training_seed':42,'seed_scope':'Official layout bank 0/1/2, exported policy inference seed unchanged','eval_num':50,'fps':25,'chunk_size':32,'execute':16,'denoise_steps':4,'history':8,'conditioning':'official_history_only','history_snapshots':2,'effective_config_sha256':plan['tasks'][task]['seeds'][str(eval_seed)]['effective_config_sha256'],'inference_seed':cfg['inference_seed'],'official_reward_and_control':True,'policy_inputs':['three_RGB','measured_native14_state','literal_instruction'],'compatibility_overlay':'None for Cover/Charger; official physics and scoring','error_policy':'fail fast; never silently replace an invalid/unstable layout','plan_sha256':hashlib.sha256((R/'status/r2-eval150-plan.json').read_bytes()).hexdigest()}
p=out/'protocol.json'
if p.exists():
 assert json.loads(p.read_text())==protocol,'Protocol/model changed; preserve existing results'
 if not (out/'COMPLETE.json').exists() and any((out/name).exists() for name in ['resume.json','result.json','IN_PROGRESS.json']):
  assert os.environ.get('FLUX_ALLOW_RESUME')=='1','Partial run requires explicit diagnosed recovery'
  result=out/'result.json';official=out/'official/_result.json';resume=out/'resume.json'
  if official.exists():
   assert result.exists() and resume.exists(),'Completed official result needs recovery reconciliation first'
   a=json.loads(result.read_text());b=json.loads(official.read_text());c=json.loads(resume.read_text())
   assert a==b and a['details']==c['details'],'Recovery manifests disagree; never overwrite valid cases'
   assert a['eval_time']==c['success_nums']+c['fail_nums']==len(c['details'])
else:p.write_text(json.dumps(protocol,indent=2))
print(json.dumps({'time':time.time(),'model_verified':True,'task':task,'checkpoint':str(export)}),flush=True)
