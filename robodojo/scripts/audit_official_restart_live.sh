R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/official-restart-live-audit.log" 2>&1
from pathlib import Path
import json,time,hashlib,csv,datetime,statistics,shutil,math
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
g=json.loads((R/'status/official-history-ready.json').read_text())
for n,h in g['files'].items():
 with (R/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,n
out=S/'runs/cover_blocks-10k-official-history-v1'
records=[]
for line in (out/'metrics.jsonl').read_text().splitlines():
 try:r=json.loads(line)
 except ValueError:continue
 if isinstance(r,dict):records.append(r)
resume=[r for r in records if r.get('event')=='resume'][-1];assert resume['time']>1791995389 and resume['step']==100 and resume['mode']=='exact'
updates=[r for r in records if 'step_time'in r and r['time']>resume['time']];assert updates[-1]['step']>100
assert all(math.isfinite(r[k]) for r in updates for k in ['loss','grad_norm'])
dec=json.JSONDecoder();ready=[]
for line in (out/'train.log').read_text(errors='replace').splitlines():
 if 'ROBODOJO_FORMAL_RANK_READY' not in line:continue
 i=line.find('{')
 try:r,end=dec.raw_decode(line[i:])
 except ValueError:continue
 ready.append(r)
assert sorted(r['rank'] for r in ready[-2:])==[0,1]
gpus={}
for row in csv.reader((out/'gpu.csv').read_text().splitlines()):
 if len(row)!=4 or not row[1].strip().isdigit():continue
 stamp=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
 if updates[-1]['time']-300<=stamp<=updates[-1]['time']:gpus.setdefault(row[1].strip(),[]).append(float(row[2].strip().split()[0]))
result={'time':time.time(),'verified':True,'source_config_sha_unchanged':True,'cover_resume':resume,'cover_ranks':ready[-2:],'cover_latest':updates[-1],'cover_gpu_mean_last300sec':{k:statistics.mean(v) for k,v in gpus.items()},'free_ssd_bytes':shutil.disk_usage(S).free,'unused_transfer_candidates':[str(p) for root in [R,S] for p in root.glob('*.incomplete')],'conveyor':'PlatformQUEUING/node0, nofreshresume orupdate; old470ishistorical. No overalltwo-task ETA untilactualstart.'}
(R/'status/official-restart-live-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
PY
