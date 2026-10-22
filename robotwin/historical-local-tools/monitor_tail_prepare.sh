B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
ps -eo pid,ppid,stat,etime,%cpu,args | grep -E 'install_tail_repair|pytest|prepare_tail_repair' | grep -v grep > "$B/status/tail-repair-processes.txt"
