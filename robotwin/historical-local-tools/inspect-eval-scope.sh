set -euo pipefail
python3 - <<'PY'
import pathlib,json,subprocess,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');pid=int((B/'status/flux-milestone-validation.pid').read_text().strip())
r={'time':time.time(),'launcher_pid':pid,'launcher':subprocess.run(['ps','-ww','-p',str(pid),'-o','pid,ppid,etime,args'],capture_output=True,text=True).stdout,'current':(B/'status/flux-milestone-validation-current.txt').read_text(),'own_processes':[]}
for line in subprocess.check_output(['ps','-eo','pid,ppid,args','--no-headers'],text=True).splitlines():
 if str(B) in line and ('policy_server' in line or 'run_eval_' in line or 'launch_flux_milestone' in line):r['own_processes'].append(line)
(B/'status/flux-10k-scope-before-stop.json').write_text(json.dumps(r,indent=2))
PY
