set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cat > "$B/scripts/convert-early.sh" <<'SH'
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
exec 9>"$B/status/conversion.lock"
flock -n 9 || exit 0
UV_CACHE_DIR=/tmp/flux-piperx-uv-cache uv run --no-project --python /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3 --with 'numpy>=2,<2.3' --with 'av>=14,<16' --with h5py --with 'pyarrow>=19,<22' --with opencv-python-headless --with pyyaml python "$B/scripts/convert_robotwin.py"
SH
nohup bash "$B/scripts/convert-early.sh" > "$B/status/conversion.log" 2>&1 < /dev/null &
echo $! > "$B/status/conversion.pid"
python3 - <<'PY'
import pathlib,json,os,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
out={}
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'uv sync --project flux-action' not in cmd:continue
  pid=p.parent;fds=[]
  for f in (pid/'fd').iterdir():
   try:
    link=os.readlink(f)
    if '/tmp/flux-piperx' in link: fds.append({'path':link,'size':f.stat().st_size})
   except OSError:pass
  out[pid.name]={'cmd':cmd,'io':(pid/'io').read_text(),'files':fds}
 except (OSError,PermissionError):pass
(B/'status/bootstrap-process.json').write_text(json.dumps(out,indent=2))
PY
