from pathlib import Path
import json,base64,shlex
root=Path(__file__).parent
B='/workspace/hdd/codex_remote_ops/flux_piperx_20261008'
names=['buffered_cpu_ema.py','train_perf_worker.py','test_buffered_ema.py','launch_perf_repair.sh']
bundle={n:base64.b64encode((root/n).read_bytes()).decode() for n in names}
code=f'''import pathlib,json,base64
B=pathlib.Path({B!r})
assert json.loads((B/'status/two-gpu-feasibility-verified.json').read_text())['save_resume_export']
R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/'runs/two-gpu-perf-repair-v1'
assert not R.exists(),'Do not overwrite an existing diagnostic'
bundle=json.loads(base64.b64decode({base64.b64encode(json.dumps(bundle).encode()).decode()!r}))
for name,data in bundle.items():
 assert '/' not in name and name.endswith(('.py','.sh'))
 (B/'scripts'/name).write_bytes(base64.b64decode(data))
'''
command=shlex.quote(B+'/flux-action/.venv/bin/python')+' -c '+shlex.quote(code)+' && bash '+B+'/scripts/launch_perf_repair.sh'
s=(root/'submit-two-gpu-feasibility.js').read_text(encoding='utf-8-sig')
start=s.index('  const j=');end=s.index('  const q=',start)
s=s[:start]+'  const ready=[{remote_preflight:"verified old feasibility; scoped source bundle installed on assigned diagnostic node"}];\n  const command='+json.dumps(command)+';\n'+s[end:]
start=s.index('      const found=');end=s.index('if(found)',start)
s=s[:start]+"      const found=a.find(x=>x.name===name&&!(x.job_id==='job-REPLACE_WITH_YOUR_JOB_ID'&&x.status==='job_failed'&&x.command.endsWith('/flux_piperx_20261008/scripts/launch_perf_repair.sh')));"+s[end:]
s=s.replace("'bash '+b+'/scripts/launch_two_gpu_feasibility.sh'",'command')
s=s.replace('job-REPLACE_WITH_YOUR_JOB_ID','job-REPLACE_WITH_YOUR_JOB_ID')
s=s.replace('FLUX 2-H100 feasibility for user-requested 10k stage: FP32 masters and both FP32 EMA profiles preserved; EMA shards on CPU; synchronize each microbatch; micro1 x2ranks x16accum global32. GPU EMA roundtrip,12 updates,checkpoint,resume13,export verification. Diagnostic only; no10k run until verified.','FLUX 2-H100 throughput repair: explicit phase timing; baseline then shared pinned asynchronous D2H snapshot for dual FP32 CPU EMA. Preserve masters/EMA math/global32. Exact dual-EMA unit check,20 updates,save,resume21,export SHA. Formal jobs stopped pending measured improvement.')
s=s.replace('return {...receipt,submissionClicked:true,url:p.url()};','return {...receipt,command:command.slice(0,110)+" ... "+command.slice(-110),submissionClicked:true,url:p.url()};')
(root/'submit-perf-repair.js').write_text(s,encoding='utf-8')
print('command characters',len(command))
