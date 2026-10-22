import pathlib,json,os,time,shutil,hashlib,collections
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
proof=json.loads((B/'status/pi05-data-verified.json').read_text())
assert (B/'status/efficiency-selected-verified.json').exists()
targets=[];preserved=B/'status/conversion-cleanup-evidence';preserved.mkdir(exist_ok=True)
for task,n in [('click_bell',3855),('handover_block',14057)]:
 source=B/f'assets/lerobot/{task}';dest=B/f'assets/pi05-v3/{task}'
 marker=json.loads((dest/'PI05_DATA_VERIFIED.json').read_text());assert marker['frames']==n and marker['episodes']==50 and proof[task]['frames']==n
 assert json.loads((dest/'meta/info.json').read_text())['total_frames']==n
 assert list((dest/'videos').rglob('*.mp4')) and list((dest/'data').rglob('*.parquet'))
 for p in [B/f'assets/pi05-v21/{task}',B/f'assets/pi05-v3/{task}_old']:
  assert p.exists() and not p.is_symlink() and p.resolve().is_relative_to(B.resolve())
  for f in p.rglob('*'):
   assert not f.is_symlink()
   if f.is_file():
    rel=f.relative_to(p)
    if rel.parts[0] in ['data','videos']:
     assert (source/rel).is_file() and os.path.samefile(f,source/rel),'Unexpected independent payload; review before delete'
  # Preserve conversion metadata and exact per-episode statistics, omit duplicate data/video payloads.
  evidence=preserved/p.parent.name/p.name;evidence.mkdir(parents=True,exist_ok=True)
  shutil.copytree(p/'meta',evidence/'meta',dirs_exist_ok=True)
  for f in p.glob('*.json'):shutil.copy2(f,evidence/f.name)
  targets.append(p)
targets += [B/'runs/cpu-ema-smoke-cpu',B/'runs/cpu-ema-smoke-cuda']
for pid in pathlib.Path('/proc').iterdir():
 if not pid.name.isdigit() or int(pid.name)==os.getpid():continue
 try:cmd=(pid/'cmdline').read_bytes().decode(errors='replace')
 except (OSError,PermissionError):continue
 assert not any(s in cmd for s in ['scripts/prepare_pi05_v21.py','scripts/convert_pi05_v30.py','scripts/test_cpu_power_ema.py']), 'Converter/unit still active'
 for fd in (pid/'fd').glob('*'):
  try:real=fd.resolve()
  except (OSError,PermissionError):continue
  assert not any(real.is_relative_to(p.resolve()) for p in targets),'Active file descriptor targets redundant directory'
receipt={'time':time.time(),'preserved_evidence':str(preserved),'deleted':[],'note':'Data/video payloads are hardlinks retained by canonical FLUX sources; apparent bytes are not physical reclaimed bytes.'}
for p in targets:
 if not p.exists():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(B.resolve())
 files=[f for f in p.rglob('*') if f.is_file()];stats=[f.stat() for f in files]
 unique={(s.st_dev,s.st_ino):s for s in stats};counts=collections.Counter((s.st_dev,s.st_ino) for s in stats)
 potentially_freed=sum(s.st_blocks*512 for key,s in unique.items() if s.st_nlink==counts[key])
 manifest=[{'relative':str(f.relative_to(p)),'bytes':s.st_size,'links':s.st_nlink} for f,s in zip(files,stats)]
 (preserved/(p.name+'-file-manifest.json')).write_text(json.dumps(manifest,indent=2))
 shutil.rmtree(p)
 receipt['deleted'].append({'path':str(p),'files':len(files),'apparent_bytes':sum(s.st_size for s in stats),'allocated_bytes_without_remaining_hardlinks':potentially_freed})
p=B/'assets/pi05-v21'
if p.exists() and not any(p.iterdir()):p.rmdir()
(B/'status/conversion-redundancy-cleanup.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
