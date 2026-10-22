set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
S=/workspace/ssd/codex_remote_ops/flux_robodojo_20261013
exec 9> "$R/status/data-controller.lock"
flock -n 9 || exit 0
for i in $(seq 1 240); do
 if [ -f "$R/status/official-v3-inspection.json" ] && [ -f "$B/status/robotwin-close-cleanup.json" ]; then
  if "$B/flux-action/.venv/bin/python" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1])).get("complete",False))' "$B/status/robotwin-close-cleanup.json"; then break; fi
 fi
 sleep 15
done
test -f "$R/status/official-v3-inspection.json"
"$B/flux-action/.venv/bin/python" -c 'import json,sys;assert json.load(open(sys.argv[1]))["complete"]' "$B/status/robotwin-close-cleanup.json"
mkdir -p "$S/hf-download-cache"
unset HF_XET_HIGH_PERFORMANCE HF_HUB_ENABLE_HF_TRANSFER HF_HUB_DISABLE_XET
export HF_XET_CACHE="$S/hf-download-cache" HF_XET_FIXED_DOWNLOAD_CONCURRENCY=8
exec "$B/flux-action/.venv/bin/python" "$R/scripts/download_selected.py"
