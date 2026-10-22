B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
ps -eo pid,etime,pcpu,wchan:24,args | grep -E '[u]pload_flux_hf.py|[s]etup_policy_server.py.*FLUX_Action|[v]erify_optimized_final.py' > "$B/status/hf-and-eval-processes.txt"
tail -c 2200 "$B/status/hf-final-upload.log" > "$B/status/hf-upload-tail.txt"
