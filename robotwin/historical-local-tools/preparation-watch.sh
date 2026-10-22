set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cat > "$B/scripts/prepare-when-ready.sh" <<'SCRIPT'
#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
exec 9>"$B/status/prepare.lock"
flock -n 9 || exit 0
for i in $(seq 1 720); do
 if [ -f "$B/status/runtime-ready.txt" ] && [ -f "$B/assets/robotwin-data.manifest.json" ]; then
   bash "$B/scripts/prepare_data.sh"
   exit
 fi
 sleep 30
done
echo PREPARATION_WAIT_EXPIRED
exit 1
SCRIPT
nohup bash "$B/scripts/prepare-when-ready.sh" > "$B/status/prepare.log" 2>&1 < /dev/null &
echo $! > "$B/status/prepare.pid"
{
 ps -eo pid,etime,pcpu,pmem,args | grep -E 'download.py|uv sync|bootstrap.sh' | grep -v grep
 du -sh "$B/assets/"* "$B/assets/".[a-z]* /tmp/flux-piperx-uv-cache 2>/dev/null || true
 tail -3 "$B/status/download.log"
} > "$B/status/transfer-progress.txt"
