"""Run the pinned RoboTwin evaluator and retain paired seeds/prompts/results."""
from pathlib import Path
import importlib.util,json,os,re,sys,time
B=Path(__file__).resolve().parents[1]
for p in [B,B/'RoboTwin',B/'XPolicyLab']:sys.path.insert(0,str(p))
spec=importlib.util.spec_from_file_location('robotwin_eval_pinned',B/'RoboTwin/scripts/eval_policy_xpolicylab.py')
upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
out=Path(os.environ['FLUX_EVAL_OUTPUT']).resolve()
assert out.is_relative_to((B/'evaluation').resolve())
out.mkdir(parents=True,exist_ok=True)
assert not (out/'result.json').exists(),'Already completed; do not repeat this evaluation'
def write(record):
    with (out/'cases.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**record})+'\n')
prepare=upstream.prepare_policy_case
def record_prepare(client,task_name,seed,instruction,action_type):
    write({'event':'case','task':task_name,'seed':seed,'instruction':instruction,'action_type':action_type})
    return prepare(client,task_name,seed,instruction,action_type)
upstream.prepare_policy_case=record_prepare
finish=upstream.notify_trial_end
def record_finish(client,task_name,seed,success):
    write({'event':'trial_end','task':task_name,'seed':seed,'success':bool(success)})
    return finish(client,task_name,seed,success)
upstream.notify_trial_end=record_finish
args=upstream.parse_args()
assert not args['eval_batch']
assert (args.get('test_num'),args['seed']) in [(1,8),(20,8),(100,0)]
assert args['env_cfg_type']=='aloha_agilex' and args['task_config']=='demo_clean'
assert args['instruction_type']=='seen' and args['action_type']=='joint'
reference_path=B/'evaluation'/args['task_name']/'flux3000-validation20/paired_seeds.json'
reference=json.loads(reference_path.read_text()) if args['seed']==8 and args['test_num']==20 else None
if reference:
    expected={x['seed']:x['instruction'] for x in reference}
    original_instruction=upstream.build_instruction
    def paired_instruction(config,episode_info,instruction_type,test_num,seed):
        if seed not in expected:raise RuntimeError(f'Paired validation seed {seed} differs from reference; investigate expert filter')
        return expected[seed]
    upstream.build_instruction=paired_instruction
args['paired_reference']=str(reference_path) if reference else None
(out/'protocol.json').write_text(json.dumps(args,indent=2))
from test_render import Sapien_TEST
Sapien_TEST()
upstream.main(args)
sys.stdout.flush()
log=Path(os.environ['FLUX_EVAL_LOG']).read_text(errors='replace')
clean=re.sub(r'\x1b\[[0-9;]*m','',log)
scores=re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>\s*[\d.]+%,\s*current seed:\s*(\d+)',clean)
assert len(scores)==args['test_num'],f'Expected {args["test_num"]} counted episodes, found {len(scores)}'
cases={x['seed']:x for x in (json.loads(s) for s in (out/'cases.jsonl').read_text().splitlines()) if x['event']=='case'}
paired=[];previous=0
for i,(successes,total,seed) in enumerate(scores,1):
    successes,total,seed=map(int,(successes,total,seed));assert total==i and successes-previous in [0,1]
    paired.append({**cases[seed],'episode':i,'success':bool(successes-previous)});previous=successes
if reference:
    assert [(x['seed'],x['instruction']) for x in paired]==[(x['seed'],x['instruction']) for x in reference], 'Paired cases changed'
(out/'paired_seeds.json').write_text(json.dumps(paired,indent=2))
(out/'result.json').write_text(json.dumps({'task':args['task_name'],'policy':args['policy_name'],'split':'validation' if args['seed']==8 else 'test','episodes':len(paired),'successes':previous,'success_rate':previous/len(paired),'complete':True,'completed_at':time.time()},indent=2))
