"""Independent functional receipt audit and scoped completed-export cleanup."""
from pathlib import Path
import csv,hashlib,json,shutil,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
S=Path(str(R).replace('/hdd/','/ssd/'))
c=json.loads((R/'status/ws-control-current.json').read_text())
assert 'job-REPLACE_WITH_YOUR_JOB_ID' in c['hostname']
assert c['ws_passed'] and c['exit_code']==0 and c['phase']=='ws_passed'
run=Path(c['run']).resolve();assert run.is_relative_to((R/'status/ws-control-jobs').resolve())
hashes={};tasks={}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for task in ['cover_blocks','match_and_pick_from_conveyor']:
 p=run/task/'ws-control-verified.json';r=json.loads(p.read_text())
 assert r['verified'] and r['control_ticks']==53 and r['layout_id']==0
 assert [x['tick'] for x in r['receipts']]==list(range(53))
 for item in r['receipts']:
  assert len(item['action_sha256'])==len(item['state_sha256'])==64
  assert set(item['rgb_sha256'])=={'cam_high','cam_left_wrist','cam_right_wrist'}
  assert all(len(v)==64 for v in item['rgb_sha256'].values())
 phases=[json.loads(x) for x in (run/task/'phases.jsonl').read_text().splitlines()]
 controls=[x for x in phases if x['phase']=='real_ws_control']
 assert [x['executed'] for x in controls]==list(range(1,54))
 assert not (run/task/'failure.json').exists()
 hashes[str(p.relative_to(R))]=sha(p);tasks[task]={'controls':53,'receipts':53,'official_executed_counters_exact':True}
events=[json.loads(x) for x in (run/'server-events.jsonl').read_text().splitlines()]
actions=[x for x in events if x['event']=='action']
assert [x['tick'] for x in actions]==list(range(53))*2
assert sum(x['tick']==52 and x['anchors']==[0,25,50] for x in actions)==2
assert events[-1]['event']=='reset' and sum(x['event']=='reset' for x in events)>=5
assert not (run/'server-failure.txt').exists()
compat=[json.loads(x) for x in (run/'match_and_pick_from_conveyor/graph-compat.jsonl').read_text().splitlines()]
post=[x for x in compat if x.get('phase')=='post_reset'];assert len(post)==1
assert post[0]['declared_speed_before']==post[0]['declared_speed_after'] and abs(post[0]['declared_speed_after']+.1)<1e-6
gpu=[]
for row in csv.reader((run/'gpu.csv').read_text().splitlines()):
 try:gpu.append(float(row[3].strip().split()[0]))
 except (ValueError,IndexError):pass
for name in ['server-events.jsonl','result.json','preflight.json']:
 hashes[str((run/name).relative_to(R))]=sha(run/name)
audit={'time':time.time(),'verified':True,'job':'job-REPLACE_WITH_YOUR_JOB_ID','tasks':tasks,'model_actions':106,'prefix_capture_and_reset':True,'peak_gpu_mib':max(gpu),'sha256':hashes,'scope':'Diagnostic export4 official WS+Isaac control interface only. 53 actions capture anchor50 while queued; offline full-prefix inference was separately verified. No task success rate or final50 evaluation.'}
(R/'status/ws-control-independent-audit.json').write_text(json.dumps(audit,indent=2))
# The platform has independently confirmed this completed job has node_count0.
# All production source configs use their own 10k run/base, never this export.
export=(S/'runs/full-preflight-v1/export-4-ema').resolve()
assert export.is_relative_to((S/'runs/full-preflight-v1').resolve())
receipt=R/'status/completed-diagnostic-export-cleanup.json'
if receipt.exists():
 assert json.loads(receipt.read_text())['verified_before_cleanup']
 raise SystemExit(0)
manifest=json.loads((export/'manifest.json').read_text())
for name,expected in manifest['sha256'].items():assert sha(export/name)==expected,name
backup=R/'status/completed-diagnostic-export-evidence';backup.mkdir(exist_ok=True)
for name in ['config.json','manifest.json']:shutil.copy2(export/name,backup/name)
model=export/'model.safetensors';size=model.stat().st_size;digest=sha(model)
evidence={'time':time.time(),'verified_before_cleanup':True,'path':str(model),'bytes':size,'sha256':digest,'audit':str(R/'status/ws-control-independent-audit.json'),'deleted':False,'production_models_or_checkpoints_deleted':False}
receipt.write_text(json.dumps(evidence,indent=2))
model.unlink()
evidence.update(deleted=True,completed=time.time());receipt.write_text(json.dumps(evidence,indent=2))
print(json.dumps({'ws_audit':True,'deleted_diagnostic_bytes':size}),flush=True)
