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
  if len(a)>1 and a[1]==str(R/'scripts/render_smoke.py').encode() and a[0].endswith(b'/python'):os.kill(int(p.name),signal.SIGTERM)
 except OSError:pass
PY
sleep 3
unset CUDA_VISIBLE_DEVICES VK_DRIVER_FILES
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$R/xdg-cache" XDG_CONFIG_HOME="$R/xdg-config" XDG_DATA_HOME="$R/xdg-data"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
nohup bash -c 'R="$1"; timeout --signal=TERM --kill-after=15s 180s "$R/sim-venv/bin/python" "$R/scripts/render_smoke.py" > "$R/status/renderer-smoke-attempt4.log" 2>&1; echo "$?" > "$R/status/renderer-smoke-attempt4.exitcode"' bash "$R" > "$R/status/renderer-launch-attempt4.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/renderer-smoke.pid"