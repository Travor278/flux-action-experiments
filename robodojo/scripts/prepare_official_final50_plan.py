"""Freeze the official layouts and final-only model selection, without running eval."""
from pathlib import Path
import ast,hashlib,json,shutil,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
assert json.loads((R/'status/ws-control-independent-audit.json').read_text())['verified']
dest=R/'XPolicyLab/policy/FLUX_RoboDojo_Official';dest.mkdir(exist_ok=True)
source=(R/'scripts/deploy_official_final50.py').read_bytes()
if (dest/'deploy.py').exists():assert (dest/'deploy.py').read_bytes()==source
else:(dest/'deploy.py').write_bytes(source)
files=['serve_official_final50.py','run_official_final50.py','check_official_final50_model.py','audit_official_final50.py','conveyor_graph_compat_final.py','launch_official_final50.sh','launch_official_final50_pair.sh']
hashes={}
for name in files:
 p=R/'scripts'/name
 if p.suffix=='.py':ast.parse(p.read_text())
 hashes[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
hashes[str((dest/'deploy.py').relative_to(R))]=hashlib.sha256(source).hexdigest()
tasks={}
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 root=R/'RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/0'
 paths=sorted(root.glob(f'{task}_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
 assert len(paths)>=50,(task,len(paths))
 tasks[task]={'training_step':10000,'eval_num':50,'layout_ids':list(range(50)),'layout_files':{i:str(p) for i,p in enumerate(paths[:50])},'layout_sha256':{i:hashlib.sha256(p.read_bytes()).hexdigest() for i,p in enumerate(paths[:50])}}
plan={'version':1,'tasks':tasks,'source_sha256':hashes,'control':'official run_eval/take_action/check success','failures':'Stop and diagnose; never replace layouts. Explicit --resume-verified required for partial results.','compatibility':'Native Isaac5.1 ReadVariable runtime overlay; original asset speed and task scoring unchanged','scope':'Prepared scripts and frozen protocol only. Final10k model load and 50 evaluation cases have NOT run.'}
p=R/'status/official-final50-plan.json'
if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(plan))
else:p.write_text(json.dumps(plan,indent=2))
(R/'status/official-final50-plan-prepared.json').write_text(json.dumps({'time':time.time(),'prepared':True,'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_files':len(hashes),'final_evaluation_started':False},indent=2))
print('FINAL50_PLAN_PREPARED_NOT_EXECUTED',flush=True)
