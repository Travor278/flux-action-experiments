R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
S=${R/\/hdd\//\/ssd\/}
export HF_XET_CACHE="$S/xet-charger-20261021" HF_XET_NUM_CONCURRENT_RANGE_GETS=8
unset HF_XET_HIGH_PERFORMANCE HF_HUB_ENABLE_HF_TRANSFER
nohup "$R/policy-venv/bin/python" "$R/scripts/download_charger_20261021.py" > "$R/status/charger-download-20261021.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/charger-download-20261021.pid"
