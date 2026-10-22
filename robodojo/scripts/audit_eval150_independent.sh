set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/scripts:$R/RoboDojo:$R/XPolicyLab:$R" PYTHONNOUSERSITE=1
"$R/sim-venv/bin/python" - <<'PY' > "$R/status/eval150-independent-audit.log" 2>&1
from pathlib import Path
import json,hashlib,time,os,subprocess
from eval150_contract import TASKS,SEEDS,official_config
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
p=R/'status/r2-eval150-plan.json';plan=json.loads(p.read_text())
for key in ['source_sha256','runtime_sha256']:
 for name,h in plan[key].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
os.environ['ROBODOJO_RUN_ID']='independent-audit-different-transport'
cases=[]
for task in TASKS:
 for seed in SEEDS:
  spec=plan['tasks'][task]['seeds'][str(seed)]
  cfg,effective=official_config(R,task,seed,54321)
  assert effective['core_sha256']==spec['effective_config_sha256']
  for i,path in spec['layout_files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==spec['layout_sha256'][i]
  assert cfg.eval_cfg.seed==seed and cfg.camera.default_frequency==25 and not cfg.task_env.get('Articulation')
  cases.append({'task':task,'seed':seed,'layouts':len(spec['layout_ids']),'config_sha':effective['core_sha256']})
from XPolicyLab.policy.FLUX_RoboDojo_R2_Eval150.deploy import eval_one_episode
negative=[]
for task in TASKS:
 env=dict(os.environ,FLUX_EVAL_SEED='0')
 r=subprocess.run([str(R/'policy-venv/bin/python'),str(R/'scripts/check_r2_eval150_model.py'),task],env=env,capture_output=True,text=True)
 assert r.returncode!=0 and 'TRAIN_COMPLETE.json' in r.stderr,(task,r.returncode,r.stderr[-1000:])
 negative.append({'task':task,'premature_model_rejected':True})
report={'time':time.time(),'verified':True,'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cases':cases,'layouts_verified':300,'premature_model_gate':negative,'runtime_simulation_executed':False,'published_baseline_calibration_executed':False,'remaining':['Actual final model loads and first full official episodes','Per-seed50 officialcounts/score/videos and3seedaggregate','No official endorsed leaderboard claim without external verification']}
(R/'status/eval150-independent-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
PY
