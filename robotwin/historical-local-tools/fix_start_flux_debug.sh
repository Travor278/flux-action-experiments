B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 "$B/scripts/normalize_xpolicy_shells.py" > "$B/status/xpolicy-shell-lf-fix.log" 2>&1
nohup bash "$B/scripts/debug_flux_rpc.sh" > "$B/status/flux-debug-launch.log" 2>&1 < /dev/null &
echo $! > "$B/status/flux-debug-launch.pid"
