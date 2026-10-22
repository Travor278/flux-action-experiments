B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
{ date -u; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader; ps -p "$(cat "$B/status/verify-flux-final.pid")" -o pid,etimes,pcpu,pmem,stat,cmd; } > "$B/status/eval-resource-check.txt"
