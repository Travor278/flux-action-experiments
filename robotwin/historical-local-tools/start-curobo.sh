set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
nohup bash "$B/scripts/bootstrap_curobo.sh" > "$B/status/curobo-bootstrap.log" 2>&1 < /dev/null &
echo $! > "$B/status/curobo-bootstrap.pid"
