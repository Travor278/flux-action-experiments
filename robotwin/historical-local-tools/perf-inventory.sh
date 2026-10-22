B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 "$B/scripts/monitor_10k.py"
python3 - <<'PY'
import pathlib,json
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
a=json.loads((B/'status/ten-k-monitor.json').read_text())
for t in a['tasks'].values():
 for k in ['tail','last_events']:t.pop(k,None)
print('PERF_INVENTORY',json.dumps(a))
PY
ps -eo pid,ppid,args | grep '[t]rain_two_gpu_worker'
