set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
if [ -f "$R/status/ws-deps-pip-install.log" ]; then cp -n "$R/status/ws-deps-pip-install.log" "$R/status/ws-deps-h100-pypi-timeout.log" || true; fi
nohup bash "$R/scripts/install_ws_deps_pip.sh" > "$R/status/ws-deps-notebook-launch.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$R/status/ws-deps-notebook.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$R/status/ws-deps-notebook-started.txt"
