B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,json,subprocess,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
r={'time':time.time(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader'],text=True),'processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),'audit_log':(B/'status/tail-final-artifact-audit.log').read_text()[-2000:]}
pid=(B/'status/tail-final-artifact-audit.pid').read_text().strip();r['audit_process']=subprocess.run(['ps','-p',pid,'-o','pid,etime,pcpu,args'],capture_output=True,text=True).stdout
(B/'status/tail-posttrain-resources.json').write_text(json.dumps(r))
PY
