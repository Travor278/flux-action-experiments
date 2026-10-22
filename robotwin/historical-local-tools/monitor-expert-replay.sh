set -euo pipefail
python3 - <<'PY'
import pathlib,json,time,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');pid=int((B/'status/expert-replay.pid').read_text());out={'time':time.time(),'pid':pid,'process':subprocess.run(['ps','-ww','-p',str(pid),'-o','pid,ppid,etime,pcpu,args'],capture_output=True,text=True).stdout,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),'gpu_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory,process_name','--format=csv,noheader'],capture_output=True,text=True).stdout,'tail':(B/'status/expert-replay-smoke.log').read_text(errors='replace')[-2500:]};(B/'status/expert-replay-monitor.json').write_text(json.dumps(out,indent=2))
PY
