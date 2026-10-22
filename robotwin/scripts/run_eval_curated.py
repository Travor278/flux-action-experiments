"""Replay expert-validated seeds exactly, retaining scored results across a partial run."""
from pathlib import Path
import importlib.util,inspect,json,os,re,sys,time,hashlib
B=Path(__file__).resolve().parents[1]
for p in [B,B/'RoboTwin',B/'XPolicyLab']:sys.path.insert(0,str(p))
spec=importlib.util.spec_from_file_location('robotwin_eval_curated',B/'RoboTwin/scripts/eval_policy_xpolicylab.py');upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
args=upstream.parse_args();assert args['seed']==8 and args['test_num']==20 and not args['eval_batch']
assert args['env_cfg_type']=='aloha_agilex' and args['action_type']=='joint' and args['task_config']=='demo_clean' and args['instruction_type']=='seen'
out=Path(os.environ['FLUX_EVAL_OUTPUT']).resolve();assert out.is_relative_to((B/'evaluation').resolve());out.mkdir(parents=True,exist_ok=True)
assert not (out/'result.json').exists(),'Do not repeat a completed validation'
log=Path(os.environ['FLUX_EVAL_LOG']);reference_path=B/f'evaluation/{args["task_name"]}/flux3000-validation20/paired_seeds.json';reference=json.loads(reference_path.read_text());assert len(reference)==20
expected={x['seed']:x['instruction'] for x in reference};assert len(expected)==20

def write(record):
 with (out/'cases.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**record})+'\n')

def scored_pairs():
 text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text(errors='replace')) if log.exists() else ''
 matches=re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>\s*[\d.]+%,\s*current seed:\s*(\d+)',text)
 rows=[json.loads(x) for x in (out/'cases.jsonl').read_text().splitlines() if x.strip()] if (out/'cases.jsonl').exists() else []
 cases={x['seed']:x for x in rows if x['event']=='case'};result={};previous_success=previous_total=0
 for successes,total,seed in matches:
  successes,total,seed=map(int,(successes,total,seed))
  if total==1:previous_success=previous_total=0
  assert total==previous_total+1 and successes-previous_success in [0,1],'Official episode counts changed unexpectedly'
  assert seed in expected and seed in cases and cases[seed]['instruction']==expected[seed]
  assert seed not in result,'Do not score a seed twice'
  result[seed]={**cases[seed],'success':bool(successes-previous_success)}
  previous_success,previous_total=successes,total
 return result

existing=scored_pairs();pending=[x['seed'] for x in reference if x['seed'] not in existing]
protocol=dict(args);protocol.update(paired_reference=str(reference_path),expert_check=False,expert_validity='reuse original expert-validated fixed seed manifest; do not resample by planner success',resumed_scored_episodes=len(existing),executed_this_attempt=len(pending))
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
if pending:
 prepare=upstream.prepare_policy_case;finish=upstream.notify_trial_end
 def record_prepare(client,task,seed,instruction,action_type):
  write({'event':'case','task':task,'seed':seed,'instruction':instruction,'action_type':action_type});return prepare(client,task,seed,instruction,action_type)
 def record_finish(client,task,seed,success):
  write({'event':'trial_end','task':task,'seed':seed,'success':bool(success)});return finish(client,task,seed,success)
 upstream.prepare_policy_case=record_prepare;upstream.notify_trial_end=record_finish
 def fixed_instruction(config,episode_info,instruction_type,test_num,seed):
  assert seed in pending,'Unexpected scene replacing a fixed validation seed'
  return expected[seed]
 upstream.build_instruction=fixed_instruction
 source=inspect.getsource(upstream.eval_remote_policy)
 needle='    while succ_seed < test_num:\n        render_freq = args["render_freq"]'
 assert source.count(needle)==1,'Pinned evaluator loop changed; inspect before adapting'
 replacement='    attempts = {}\n    while succ_seed < test_num:\n        now_seed = fixed_seed_list[succ_seed]\n        attempts[now_seed] = attempts.get(now_seed, 0) + 1\n        if attempts[now_seed] > 3:raise RuntimeError(f"Fixed validation seed {now_seed} could not run after 3 attempts")\n        render_freq = args["render_freq"]'
 upstream.fixed_seed_list=pending;exec(compile(source.replace(needle,replacement),'<fixed-expert-validated-seed-loop>','exec'),upstream.__dict__)
 (out/'replay-implementation.json').write_text(json.dumps({'time':time.time(),'pending_seeds':pending,'retained_seeds':list(existing),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'change':'replace sequential seed advancement with fixed manifest indexing; capped setup retries; official policy success loop unchanged'},indent=2))
 print('CURATED_REPLAY_BEGIN '+json.dumps({'time':time.time(),'retained':len(existing),'pending':pending}),flush=True)
 from test_render import Sapien_TEST
 Sapien_TEST();args['expert_check']=False;args['test_num']=len(pending);upstream.main(args);sys.stdout.flush()
pairs=scored_pairs();assert set(pairs)==set(expected),'Incomplete fixed validation'
ordered=[{**pairs[x['seed']],'episode':i} for i,x in enumerate(reference,1)]
(out/'paired_seeds.json').write_text(json.dumps(ordered,indent=2))
result={'task':args['task_name'],'policy':args['policy_name'],'split':'validation','episodes':20,'successes':sum(x['success'] for x in ordered),'success_rate':sum(x['success'] for x in ordered)/20,'complete':True,'completed_at':time.time(),'expert_validity':'fixed expert-validated scene replay','retained_partial_episodes':len(existing)}
(out/'result.json').write_text(json.dumps(result,indent=2));print('FIXED_PAIRED_VALIDATION_COMPLETE '+json.dumps(result),flush=True)
