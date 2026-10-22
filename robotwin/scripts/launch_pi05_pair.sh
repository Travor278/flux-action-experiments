#!/bin/bash
set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
for TASK in click_bell handover_block; do
  if [ -s "$B/status/pi05_${TASK}_seed42/COMPLETE.json" ]; then
    echo "PI05_ALREADY_COMPLETE $TASK"
    continue
  fi
  bash "$B/scripts/launch_pi05.sh" "$TASK"
done
printf 'BOTH_PI05_TRAINING_COMPLETE\n' > "$B/status/pi05-pair-complete.txt"
