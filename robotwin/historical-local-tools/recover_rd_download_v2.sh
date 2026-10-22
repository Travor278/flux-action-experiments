set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$R/policy-venv/bin/python" -m pytest "$R/scripts/test_download_failure.py" -q > "$R/status/download-failure-test.log" 2>&1
cp "$R/status/selected-download-progress.json" "$R/status/selected-download-progress-attempt1.json"
cp "$R/status/selected-download.log" "$R/status/selected-download-attempt1.log"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import os,signal,time,json
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');p=Path('/proc/17679')
if p.exists():
 a=(p/'cmdline').read_bytes().split(b'\0');assert a[1]==str(R/'scripts/download_selected.py').encode();os.kill(17679,signal.SIGTERM)
 for _ in range(50):
  if not p.exists():break
  time.sleep(.1)
 assert not p.exists(),'Previous writer still alive'
(R/'status/download-recovery.json').write_text(json.dumps({'time':time.time(),'old_pid':17679,'reason':'py-spy confirms main thread stuck in executor __exit__/shutdown after a per-file exception; workers still downloading while progress not updated','first_exception':'not yet emitted; preserve stack/log/cache','new_behavior':'bounded pending work, per-file retry/error capture and durable SHA receipts','source_and_transport_unchanged':True},indent=2))
PY
unset HF_XET_HIGH_PERFORMANCE HF_HUB_ENABLE_HF_TRANSFER HF_HUB_DISABLE_XET
export HF_XET_CACHE=/workspace/ssd/codex_remote_ops/flux_robodojo_20261013/hf-download-cache HF_XET_FIXED_DOWNLOAD_CONCURRENCY=8
nohup "$B/flux-action/.venv/bin/python" "$R/scripts/download_selected_v2.py" > "$R/status/selected-download-v2.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/selected-download.pid"