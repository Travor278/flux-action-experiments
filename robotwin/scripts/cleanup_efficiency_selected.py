import pathlib,json,shutil,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/'runs/two-gpu-efficiency-sweep-v1/selected-validation'
proof=json.loads((B/'status/efficiency-selected-verified.json').read_text());assert all(proof[k] for k in ['complete','save_resume_export','safe_stop_verified'])
assert json.loads((R/'step-5/state.json').read_text())['step']==5 and (R/'step-5/COMPLETE').exists()
assert pathlib.Path(proof['validation_run'])==R
dest=B/'status/efficiency-selected-evidence';dest.mkdir(exist_ok=True)
for rel in ['step-5/state.json','step-5/COMPLETE','step-5/config.json','export-5-ema/manifest.json','export-5-ema/config.json','STOPPED_SAFELY.json']:
 p=R/rel
 if p.exists():shutil.copy2(p,dest/rel.replace('/','__'))
receipt={'time':time.time(),'selected':proof['selected'],'deleted':[]}
for name in ['step-5','export-5-ema']:
 p=R/name;assert p.resolve().parent==R.resolve() and not p.is_symlink()
 size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file());shutil.rmtree(p)
 receipt['deleted'].append({'path':str(p),'bytes':size})
(B/'status/efficiency-selected-cleanup.json').write_text(json.dumps(receipt,indent=2))
