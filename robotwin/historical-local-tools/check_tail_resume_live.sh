B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,subprocess,json,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');p=(B/'status/flux-tail-test.pid').read_text().strip()
r={'time':time.time(),'pid':p,'process':subprocess.run(['ps','-p',p,'--ppid',p,'-o','pid,ppid,etime,pcpu,args'],capture_output=True,text=True).stdout,'launchlog':(B/'status/flux-tail-handover-resume-launch.log').read_text()[-1200:],'current':(B/'status/flux-tail-test-current.txt').read_text()}
(B/'status/tail-resume-live.json').write_text(json.dumps(r))
PY
