R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
ps -eo pid,etime,pcpu,pmem,args | grep -E '[p]repare_official_history.py|[f]fmpeg' | head -20 > "$R/status/official-prepare-process.txt"
