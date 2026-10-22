B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,json,time,subprocess,os
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');r={'time':time.time(),'uptime':pathlib.Path('/proc/uptime').read_text(),'logs':{}}
for n in ['flux-tail-final-test-launch.log','flux-tail-10k-test100-handover_block.log','flux-tail-10k-test100-handover_block-server.log']:
 r['logs'][n]=(B/'status'/n).read_text(errors='replace')[-7500:]
r['processes']=subprocess.check_output(['ps','-eo','pid,ppid,lstart,args'],text=True);r['processes']='\n'.join(x for x in r['processes'].splitlines() if str(B) in x)
r['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
(B/'status/tail-handover-interruption.json').write_text(json.dumps(r,indent=2))
PY
