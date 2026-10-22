set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,json,time,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');r={'time':time.time()}
p=B/'status/hf-final-upload.pid'
if p.exists():
 pid=int(p.read_text());r['pid']=pid;r['process_exists']=pathlib.Path(f'/proc/{pid}').exists();r['process']=subprocess.run(['ps','-ww','-p',str(pid),'-o','pid,etime,pcpu,wchan:24,args'],capture_output=True,text=True).stdout
 try:r['io']=pathlib.Path(f'/proc/{pid}/io').read_text()
 except OSError:pass
p=B/'status/hf-final-uploads.json';r['status']=json.loads(p.read_text()) if p.exists() else None
p=B/'status/hf-final-upload.log';s=p.read_text(errors='replace') if p.exists() else '';r['tail']=s[-2000:]
if 'Traceback' in s:r['error']=s[s.index('Traceback'):][:5000]
(B/'status/hf-upload-monitor.json').write_text(json.dumps(r,indent=2))
PY
