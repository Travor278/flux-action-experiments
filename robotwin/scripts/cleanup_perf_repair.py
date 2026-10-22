import pathlib,json,shutil,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/'runs/two-gpu-perf-repair-v1'
proof=json.loads((B/'status/two-gpu-perf-repair-result.json').read_text());assert proof['candidate_valid'] and proof['save_resume_export']
assert json.loads((R/'step-21/state.json').read_text())['step']==21 and (R/'step-21/COMPLETE').exists()
dest=B/'status/perf-repair-evidence';dest.mkdir(exist_ok=True)
for rel in ['step-21/state.json','step-21/COMPLETE','step-21/config.json','export-21-ema/manifest.json','export-21-ema/config.json']:
 p=R/rel
 if p.exists():shutil.copy2(p,dest/rel.replace('/','__'))
receipt={'time':time.time(),'proof':proof,'deleted':[]}
for name in ['step-21','export-21-ema']:
 p=R/name;assert p.resolve().parent==R.resolve() and not p.is_symlink()
 size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file());shutil.rmtree(p)
 receipt['deleted'].append({'path':str(p),'bytes':size})
(B/'status/perf-repair-cleanup.json').write_text(json.dumps(receipt,indent=2))
