B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,json,time,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');r={'time':time.time(),'processes':{}}
for pid in [4049716,4076687]:
 p=pathlib.Path('/proc')/str(pid)
 r['processes'][pid]={'alive':p.exists(),'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace') if p.exists() else None,'stat':(p/'stat').read_text() if p.exists() else None}
for task in ['click_bell','handover_block']:
 p=B/f'evaluation/{task}/flux-10k-test100/result.json';r[task]=json.loads(p.read_text()) if p.exists() else None
r['complete_marker']=(B/'status/flux-final-test-complete.txt').exists()
r['log_tail']=(B/'status/flux-10k-test100-handover_block.log').read_text(errors='replace')[-1500:]
r['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
(B/'status/old-eval-process-audit.json').write_text(json.dumps(r,indent=2))
PY
