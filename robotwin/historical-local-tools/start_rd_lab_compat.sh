R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup "$R/sim-venv/bin/python" "$R/scripts/install_isaaclab_compat.py" > "$R/status/isaaclab-compat-install.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/isaaclab-compat-install.pid"