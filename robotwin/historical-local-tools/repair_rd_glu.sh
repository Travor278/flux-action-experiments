set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import os,signal,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');stopped=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if len(args)>1 and args[1]==str(R/'scripts/render_smoke.py').encode() and args[0].endswith(b'/python'):
   stopped.append(int(p.name));os.kill(int(p.name),signal.SIGTERM)
 except OSError:pass
(R/'status/renderer-first-failure.json').write_text(json.dumps({'time':time.time(),'rootcause':'libGLU.so.1 missing from dynamic loader','stopped_own_python_pids':stopped},indent=2))
PY
sleep 3
cp "$R/status/renderer-smoke.log" "$R/status/renderer-smoke-attempt1.log"
mkdir -p "$R/system-libs/packages" "$R/system-libs/root"
cd "$R/system-libs/packages"
apt-get download libglu1-mesa libopengl0 > "$R/status/glu-library-install.log" 2>&1
for p in "$R"/system-libs/packages/*.deb; do dpkg-deb -x "$p" "$R/system-libs/root"; done
LD_LIBRARY_PATH="$R/system-libs/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" ldd "$R/system-libs/root/usr/lib/x86_64-linux-gnu/libGLU.so.1" >> "$R/status/glu-library-install.log" 2>&1