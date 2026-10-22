B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,os,json,time,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'roots':[]}
for root in [B/'runs',S/'runs']:
 if not root.exists():continue
 for run in sorted(root.iterdir()):
  if not run.is_dir() or run.is_symlink():continue
  children=[]
  for p in sorted(run.iterdir()):
   if p.is_dir():
    logical=allocated=files=0
    for folder,dirs,names in os.walk(p,followlinks=False):
     dirs[:]=[x for x in dirs if not pathlib.Path(folder,x).is_symlink()]
     for n in names:
      q=pathlib.Path(folder,n)
      if q.is_symlink():continue
      st=q.stat();logical+=st.st_size;allocated+=st.st_blocks*512;files+=1
    children.append({'path':str(p),'bytes':logical,'allocated':allocated,'files':files,'complete':(p/'COMPLETE').exists()})
  out['roots'].append({'run':str(run),'directories':children,'files':[x.name for x in run.iterdir() if x.is_file()]})
out['ssd_free']=shutil.disk_usage(S).free
(B/'status/failed-training-cleanup-inventory.json').write_text(json.dumps(out,indent=2))
PY
