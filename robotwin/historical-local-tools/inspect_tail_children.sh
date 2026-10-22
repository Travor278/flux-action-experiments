B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
ps -p 2206877 --ppid 2206877 -o pid,ppid,stat,etime,%cpu,wchan:24,args > "$B/status/tail-children.txt"
wc -c "$B/status/tail-repair-tests.log" "$B/status/tail-repair-prepare.log" >> "$B/status/tail-children.txt" 2>&1
