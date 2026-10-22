import pathlib,json,time,shutil,os,fcntl
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
proof=json.loads((B/'status/hf-checkpoint-only-verified.json').read_text());assert proof['complete'] and len(proof['tasks'])==2
status=json.loads((B/'status/hf-final-uploads.json').read_text());assert status['complete']
pid=int((B/'status/hf-final-upload.pid').read_text());proc=pathlib.Path(f'/proc/{pid}')
if proc.exists():assert str(B/'scripts/upload_flux_hf.py') not in (proc/'cmdline').read_bytes().decode(errors='replace'),'Active uploader'
lock=(B/'status/hf-final-upload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt={'time':time.time(),'hub_revisions':{t:r['revision'] for t,r in proof['tasks'].items()},'deleted':[],'source_exports_preserved':True}
for task in ['click_bell','handover_block']:
 src=S/f'runs/{task}-10k-opt-v1/export-10000-ema';stage=S/'hf-publish'/task
 assert src.is_dir() and stage.resolve()==S.resolve()/'hf-publish'/task
 for name in ['model.safetensors','config.json','manifest.json']:assert os.path.samefile(src/name,stage/name)
for p in [S/'hf-publish',S/'hf-transfer-cache']:
 if not p.exists():continue
 assert not p.is_symlink() and p.resolve().parent==S.resolve()
 files=[f for f in p.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in files)
 reclaim=sum(f.stat().st_blocks*512 for f in files if f.stat().st_nlink==1)
 shutil.rmtree(p);receipt['deleted'].append({'path':str(p),'files':len(files),'apparent_bytes':size,'allocated_bytes_without_remaining_hardlinks':reclaim})
p=B/'assets/hf-publish-transfer'
if p.is_symlink():assert p.readlink()==S/'hf-publish';p.unlink()
(B/'status/hf-upload-staging-cleanup.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
