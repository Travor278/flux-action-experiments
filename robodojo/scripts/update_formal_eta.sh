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
