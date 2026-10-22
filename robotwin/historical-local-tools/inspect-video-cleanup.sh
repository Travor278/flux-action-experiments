set -euo pipefail
python3 - <<'PY'
import pathlib,json,time,os
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'candidate_dirs':[],'source_video_root':str(B/'RoboTwin/eval_result/click_bell/FLUX_Action/demo_clean/flux-10k-test100'),'active':[]}
for root in [S/'runs',B/'runs']:
 if root.exists():
  for run in root.iterdir():
   if run.is_dir():
    for d in run.iterdir():
     if d.is_dir() and (d.name.startswith('step-') or d.name=='checkpoints'):
      files=[f for f in d.rglob('*') if f.is_file()];out['candidate_dirs'].append({'path':str(d),'bytes':sum(f.stat().st_size for f in files),'files':len(files)})
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  c=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(B) in c or str(S) in c:out['active'].append({'pid':int(p.parent.name),'cmd':c[:600]})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
p=B/'evaluation/click_bell/flux-10k-test100/paired_seeds.json';pairs=json.loads(p.read_text());out['video_cases']=[{'episode':x['episode']-1,'seed':x['seed'],'success':x['success'],'instruction':x['instruction']} for x in pairs if x['success'] or x['episode']-1 in [0,1,18,20,99]]
root=pathlib.Path(out['source_video_root']);out['video_dir']=str(next(root.glob('*/episode0.mp4')).parent)
(B/'status/flux-video-cleanup-inspection.json').write_text(json.dumps(out,indent=2))
PY
