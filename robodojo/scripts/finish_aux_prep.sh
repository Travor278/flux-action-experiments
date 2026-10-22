set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import subprocess,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');rows=[]
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 p=subprocess.run([str(R/'policy-venv/bin/python'),str(R/'scripts/check_final50_model.py'),task],capture_output=True,text=True)
 assert p.returncode!=0 and 'TRAIN_COMPLETE.json' in p.stderr,p.stderr
 rows.append({'task':task,'premature_evaluation_rejected':True,'reason':'Final10k training marker absent'})
(R/'status/final50-premature-launch-guard.json').write_text(json.dumps({'time':time.time(),'verified':True,'tasks':rows,'scope':'Expected refusal before final model exists; no evaluation run'},indent=2))
PY
"$R/policy-venv/bin/python" "$R/scripts/monitor_formal_training.py"
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/formal-full-eta.log" 2>&1
from pathlib import Path
import json,time,math,datetime
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');m=json.loads((R/'status/formal-training-monitor.json').read_text());now=time.time();rows={}
for name,v in m['tasks'].items():
 r=v['latest'];rate=v['stable_seconds_per_update'];assert r['step']>=250 and rate and rate>0
 end=r['time']+(10000-r['step'])*rate+1800
 rows[name]={'step':r['step'],'last_update_time':r['time'],'post220_seconds_per_update':rate,'forecast_end_time':end,'remaining_seconds':max(0,end-now),'save_export_reserve_seconds':1800}
latest=max(v['forecast_end_time'] for v in rows.values());minutes=math.ceil(max(0,latest-now)/60)
result={'time':now,'tasks':rows,'full_training_remaining_minutes':minutes,'later_forecast_beijing':datetime.datetime.fromtimestamp(latest,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'Bothruns post220 measured normal loginterval medians +1800s eachsave/export allowance. Forecast notguarantee; nearest pendingevaluationprep mayrequire earlier supervision.'}
(R/'status/formal-full-training-eta.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
PY
