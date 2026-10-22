R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
ps -eo pid,etime,pcpu,pmem,args | grep '[a]udit_official_history_restart.py' > "$R/status/official-restart-audit-process.txt"
