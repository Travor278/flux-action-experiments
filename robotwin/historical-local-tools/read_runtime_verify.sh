B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
ps -eo pid,etime,args | grep '[v]erify_optimized_final.py' > "$B/status/optimized-final-verify-process.txt"
ls -l "$B/status/optimized-final-artifacts-verified.json" >> "$B/status/optimized-final-verify-process.txt" 2>&1
