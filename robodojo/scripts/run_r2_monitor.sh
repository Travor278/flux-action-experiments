R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" "$R/scripts/monitor_official_r2.py" > "$R/status/r2-monitor-refresh.log" 2>&1
ps -eo pid,etime,pcpu,args | grep -E '[p]repare_official_r2.py|[d]ownload_charger_20261021.py' > "$R/status/r2-preparation-process.txt"
