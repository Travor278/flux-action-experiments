B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,os,json,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'candidates':[],'roots':{}}
for root in [B,S]:
 out['roots'][str(root)]=[p.name for p in root.iterdir() if p.is_dir()]
 for branch in ['runs','checkpoints','tmp','profile','checkpoint-repair','artifacts']:
  start=root/branch
  if not start.exists():continue
  for folder,dirs,files in os.walk(start,followlinks=False):
   dirs[:]=[d for d in dirs if not pathlib.Path(folder,d).is_symlink()]
   p=pathlib.Path(folder)
   if any(x in p.name.lower() for x in ['step-','optimizer','checkpoint','incomplete','partial']):out['candidates'].append(str(p))
(B/'status/failed-checkpoint-final-scan.json').write_text(json.dumps(out,indent=2))
PY
