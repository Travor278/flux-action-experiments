from pathlib import Path
import json,os,time,shutil
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
roots=[B/'assets',B/'runs',B/'status',B/'uv-cache',Path('/tmp/flux-piperx-xet'),Path('/tmp/flux-piperx-eval-xet')]
items=[]
for root in roots:
 if not root.exists():continue
 for p in ([root] if str(root).startswith('/tmp/') else root.iterdir()):
  if p.is_symlink():continue
  size=0;allocated=0;files=0
  for f in ([p] if p.is_file() else p.rglob('*')):
   if f.is_file() and not f.is_symlink():
    st=f.stat();size+=st.st_size;allocated+=st.st_blocks*512;files+=1
  items.append({'path':str(p),'bytes':size,'allocated_bytes':allocated,'files':files})
status={'time':time.time(),'items':sorted(items,key=lambda x:x['allocated_bytes'],reverse=True),'ssd_free_bytes':shutil.disk_usage(S).free,'training':{}}
for task in ['click_bell','handover_block']:
 p=S/'runs'/task
 status['training'][task]={'log_tail':(p/'train.log').read_text(errors='replace')[-5000:] if (p/'train.log').exists() else None,'metrics_tail':(p/'metrics.jsonl').read_text()[-5000:] if (p/'metrics.jsonl').exists() else None}
(B/'status/cleanup-audit.json').write_text(json.dumps(status,indent=2))
