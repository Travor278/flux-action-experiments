from pathlib import Path
import hashlib,json,os,shutil,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
S=Path(str(R).replace('/hdd/','/ssd/'))
out=S/'runs/full-preflight-v1'; target=out/'step-4'
receipt=R/'status/completed-preflight-cleanup.json'
if receipt.exists():
    print(receipt.read_text());raise SystemExit(0)
audit=json.loads((R/'status/full-preflight-export-audit.json').read_text())
current=json.loads((R/'status/export-inference-current.json').read_text())
assert audit['verified'] and current['verified'] and current['exit_code']==0
assert current['hostname']=='job-REPLACE_WITH_YOUR_JOB_ID-worker-0'
run=Path(current['run']).resolve();assert run.is_relative_to((R/'status/export-inference-jobs').resolve())
report=run/'export-inference-verified.json'; proof=json.loads(report.read_text())
assert target.resolve()==S.resolve()/'runs/full-preflight-v1/step-4'
assert target.is_dir() and not target.is_symlink() and (target/'COMPLETE').is_file()
assert json.loads((target/'state.json').read_text())['step']==4
# Own notebook processes only; both remote compute jobs were independently confirmed SUCCESS/node0.
active=[]
for proc in Path('/proc').glob('[0-9]*'):
    if int(proc.name)==os.getpid():continue
    try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (OSError,PermissionError):continue
    if str(out) in cmd or ('probe_export_inference.py' in cmd and str(R) in cmd):active.append({'pid':proc.name,'cmd':cmd[:500]})
assert not active,active
evidence=R/'status/completed-preflight-cleanup-evidence';evidence.mkdir(exist_ok=True)
files=[];total=0
for p in target.rglob('*'):
    assert not p.is_symlink(),str(p)
    if not p.is_file():continue
    rel=p.relative_to(target);size=p.stat().st_size;total+=size
    item={'path':str(rel),'bytes':size}
    if size<16*1024*1024:
        dest=evidence/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256']
    files.append(item)
plan={'time':time.time(),'target':str(target),'files':files,'bytes':total,'inference_report':str(report),'inference_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'export_retained':str(out/'export-4-ema'),'reason':'Completed diagnostic training state; exact GPU restore and real exported GPU inference passed. Export retained for simulator adapter integration.'}
(evidence/'manifest.json').write_text(json.dumps(plan,indent=2))
shutil.rmtree(target)
assert not target.exists()
plan.update(deleted=True,finished=time.time(),free_bytes=shutil.disk_usage(S).free)
receipt.write_text(json.dumps(plan,indent=2))
print(json.dumps({k:v for k,v in plan.items() if k!='files'},indent=2))
