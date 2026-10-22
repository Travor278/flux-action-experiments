set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
if [ -s "$B/assets/flux-base.manifest.json" ]; then exit 0; fi
python3 - <<'PY'
import pathlib,os,signal,time,json
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
target=str(B/'scripts/download.py').encode()
pids=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  if target in p.read_bytes().split(b'\0'):
   pid=int(p.parent.name);pids.append(pid);os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
deadline=time.time()+20
while any(pathlib.Path(f'/proc/{p}').exists() for p in pids) and time.time()<deadline:time.sleep(.25)
live=[]
for pid in pids:
 p=pathlib.Path(f'/proc/{pid}/cmdline')
 if p.exists() and target in p.read_bytes().split(b'\0'):live.append(pid)
if live:raise RuntimeError(f'Old downloader has not stopped: {live}')
(B/'status/download-tuning.json').write_text(json.dumps({'time':time.time(),'stopped_owned_pids':pids,'baseline_bytes_per_second':1000000,'new_fixed_concurrency':8,'reason':'bounded throughput tuning after sustained low throughput; preserve exact revision, local_dir and Xet cache'},indent=2))
PY
printf '\nControlled resume: fixed concurrency8; no high-performance override\n' >> "$B/status/download.log"
nohup env -u HF_XET_HIGH_PERFORMANCE HF_XET_FIXED_DOWNLOAD_CONCURRENCY=8 HF_XET_CACHE=/tmp/flux-piperx-xet "$B/flux-action/.venv/bin/python" "$B/scripts/download_fast.py" >> "$B/status/download.log" 2>&1 < /dev/null &
echo $! > "$B/status/download.pid"
