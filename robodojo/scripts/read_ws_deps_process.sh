R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
ps -eo pid,etime,state,args | grep -F 'flux_robodojo_20261013/ws-deps' | grep -v 'grep -F' > "$R/status/ws-deps-process.txt" || true
