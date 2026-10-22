import pathlib,json,shutil,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/'runs/two-gpu-feasibility-v1'
proof=json.loads((B/'status/two-gpu-feasibility-verified.json').read_text());assert proof['verified'] and proof['save_resume_export']
assert json.loads((R/'step-13/state.json').read_text())['step']==13 and (R/'step-13/COMPLETE').exists()
dest=B/'status/two-gpu-success';dest.mkdir(exist_ok=True)
for rel in ['step-13/state.json','step-13/COMPLETE','step-13/config.json','export-13-ema/manifest.json','export-13-ema/config.json']:
 p=R/rel
 if p.exists():shutil.copy2(p,dest/(rel.replace('/','__')))
receipt={'time':time.time(),'proof':proof,'deleted':[]}
for name in ['step-13','export-13-ema']:
 p=R/name;assert p.resolve().parent==R.resolve() and not p.is_symlink()
 size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file());shutil.rmtree(p)
 receipt['deleted'].append({'path':str(p),'bytes':size})
(B/'status/two-gpu-success-cleanup.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
