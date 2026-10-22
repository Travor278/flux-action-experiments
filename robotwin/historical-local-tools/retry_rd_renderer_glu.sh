set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import os,signal
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
for p in Path('/proc').glob('[0-9]*'):
 try:
  a=(p/'cmdline').read_bytes().split(b'\0')
  if len(a)>=4 and a[0].endswith(b'apt-get') and a[1:4]==[b'download',b'libglu1-mesa',b'libopengl0'] and (p/'cwd').resolve()==R/'system-libs/packages':os.kill(int(p.name),signal.SIGTERM)
 except OSError:pass
PY
export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1
ldd "$B/system-libs/usr/lib/x86_64-linux-gnu/libGLU.so.1" > "$R/status/glu-reused-ldd.txt"
if grep -q 'not found' "$R/status/glu-reused-ldd.txt"; then exit 21; fi
nohup bash -c 'R="$1"; timeout --signal=TERM --kill-after=15s 300s "$R/sim-venv/bin/python" "$R/scripts/render_smoke.py" > "$R/status/renderer-smoke-attempt2.log" 2>&1; echo "$?" > "$R/status/renderer-smoke-attempt2.exitcode"' bash "$R" > "$R/status/renderer-launch-attempt2.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/renderer-smoke.pid"