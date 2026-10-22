set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
if [ -s "$B/status/curobo-import-check.txt" ]; then exit 0; fi
pid=$(cat "$B/status/curobo-bootstrap.pid")
if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$B/scripts/bootstrap_curobo_v2.sh"; then
 echo "Owned curobo setup still running" > "$B/status/curobo-launch-check.txt"
else
 nohup bash "$B/scripts/bootstrap_curobo_v2.sh" >> "$B/status/curobo-bootstrap-v2.log" 2>&1 < /dev/null &
 echo $! > "$B/status/curobo-bootstrap.pid"
 echo "Restarted after old watcher child released inherited lock" > "$B/status/curobo-launch-check.txt"
fi
