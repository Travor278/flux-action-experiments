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
