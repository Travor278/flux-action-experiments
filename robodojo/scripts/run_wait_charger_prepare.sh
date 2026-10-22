set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/flux-action-official-v1/src:$R/scripts" PYTHONNOUSERSITE=1
nohup bash -c '
exec 9>"$1/status/charger-prepare-20261021.lock"
flock -n 9 || exit 3
for i in $(seq 1 360); do
 if [ -f "$1/status/charger-download-verified-20261021.json" ]; then
  "$1/policy-venv/bin/python" "$1/scripts/prepare_official_r2.py" plug_in_charger
  rc=$?; echo "$rc" > "$1/status/charger-prepare-20261021.exitcode"; exit "$rc"
 fi
 pid=$(cat "$1/status/charger-download-20261021.pid")
 if ! kill -0 "$pid" 2>/dev/null; then echo DOWNLOAD_EXITED_WITHOUT_VERIFIED; exit 4; fi
 sleep 20
done
exit 124
' _ "$R" > "$R/status/official-r2-plug_in_charger-prepare.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/charger-prepare-20261021.pid"
