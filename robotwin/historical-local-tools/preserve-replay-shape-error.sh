set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
mkdir -p "$B/evaluation/expert-replay-click/startup-errors"
cp "$B/status/expert-replay-smoke.log" "$B/status/expert-replay-startup-attempt2.log"
if test -s "$B/status/expert-replay-failure.json";then mv "$B/status/expert-replay-failure.json" "$B/evaluation/expert-replay-click/startup-errors/failure-attempt2.json";fi
if test -s "$B/evaluation/expert-replay-click/episode-0/error.json";then mv "$B/evaluation/expert-replay-click/episode-0/error.json" "$B/evaluation/expert-replay-click/startup-errors/episode0-attempt2.json";fi
