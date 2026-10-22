R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import os,json,time,subprocess
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');r={'time':time.time(),'driver_capabilities':os.environ.get('NVIDIA_DRIVER_CAPABILITIES'),'driver_libraries':{},'own_renderer_processes':[]}
for name in ['libnvidia-rtcore.so*','libnvoptix.so*','libnvidia-glcore.so*','libGLX_nvidia.so*']:
 r['driver_libraries'][name]=[str(p) for root in [Path('/usr/lib/x86_64-linux-gnu'),Path('/usr/local/nvidia/lib64')] for p in root.glob(name)]
for p in Path('/proc').glob('[0-9]*'):
 try:
  a=(p/'cmdline').read_bytes().split(b'\0')
  if len(a)>1 and a[1] in [str(R/'scripts/render_smoke.py').encode(),str(R/'scripts/render_smoke_lab.py').encode()]:r['own_renderer_processes'].append(int(p.name))
 except OSError:pass
r['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
(R/'status/renderer-runtime-audit.json').write_text(json.dumps(r,indent=2))
PY