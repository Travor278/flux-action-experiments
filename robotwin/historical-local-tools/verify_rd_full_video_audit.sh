set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
from collections import Counter
import json,time,hashlib,subprocess
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
ledger=R/'status/all-video-decode-receipts.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines()]
marker=json.loads((R/'status/all-selected-videos-decoded.json').read_text())
assert marker['complete'] and len(rows)==600 and (R/'status/all-video-decode.exitcode').read_text().strip()=='0'
keys=[(v['task'],v['episode'],v['camera']) for v in rows];assert len(set(keys))==600
for task,ids in [('cover_blocks',range(300,400)),('match_and_pick_from_conveyor',range(1600,1700))]:
 assert {k for k in keys if k[0]==task}=={(task,i,c) for i in ids for c in ['cam_high','cam_left_wrist','cam_right_wrist']}
for v in rows:
 assert v['decoded']==v['count'] and v['metadata_sha256']==marker['metadata_sha256']
 assert len(v['rgb_samples'])==3 and len({s['index'] for s in v['rgb_samples']})==3
assert sum(v['count'] for v in rows)==292682
anomalies=[{'task':v['task'],'episode':v['episode'],'camera':v['camera'],'missing':v['numeric_length']-v['count']} for v in rows if v['numeric_length']!=v['count']]
assert len(anomalies)==4 and sum(v['missing'] for v in anomalies)==4
processes=subprocess.check_output(['ps','-ww','-eo','pid,args'],text=True)
own=[line for line in processes.splitlines() if str(R/'scripts/audit_all_selected_videos.py') in line]
assert not own,own
out={'time':time.time(),'verified':True,'spans':len(rows),'real_frames':292682,'per_task':{task:{'spans':sum(v['task']==task for v in rows),'real_frames':sum(v['count'] for v in rows if v['task']==task)} for task in ['cover_blocks','match_and_pick_from_conveyor']},'known_missing_camera_frames':anomalies,'ledger_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),'process_released':True,'scope':'Independent coverage/unique keys/source metadata/frame counts/RGB sample receipts audit, not cue or policy score'}
(R/'status/all-video-decode-independent-audit.json').write_text(json.dumps(out,indent=2))
print('FULL_VIDEO_AUDIT_INDEPENDENT_PASS')
PY