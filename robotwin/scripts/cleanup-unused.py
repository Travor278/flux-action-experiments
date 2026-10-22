from pathlib import Path
import json,time,os
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008').resolve()
manifest=json.loads((B/'assets/flux-base.manifest.json').read_text())
root=B/'assets/flux-base'
for f in manifest['files']:
 p=root/f['path'];assert p.is_file() and p.stat().st_size==f['bytes'],f['path']
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try: cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except (OSError,PermissionError):continue
 if str(B/'scripts/download.py') in cmd or str(B/'scripts/download_fast.py') in cmd:raise RuntimeError('download still active')
candidates=list((root/'.cache').rglob('*.incomplete'))
# Transport-only archives have local source copies and deployed pinned contents.
for name,deployed in [('robotwin-code.tar.gz','RoboTwin/scripts'),('flux-eval-adapter.tar.gz','XPolicyLab/policy/FLUX_Action'),('xpolicy-utils.tar.gz','XPolicyLab/utils')]:
 p=B/'assets'/name
 if p.is_file() and (B/deployed).is_dir():candidates.append(p)
for p in (B/'status').iterdir():
 if p.name.endswith('\r') and p.is_file() and (p.with_name(p.name.rstrip('\r'))).is_file():candidates.append(p)
removed=[]
for p in candidates:
 assert not p.is_symlink() and p.resolve().is_relative_to(B) and p.is_file(),str(p)
 st=p.stat();removed.append({'path':str(p),'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'links_before':st.st_nlink});p.unlink()
result={'time':time.time(),'manifest_revision':manifest['revision'],'preserved_verified_model_files':len(manifest['files']),'removed':removed,'logical_bytes_removed':sum(x['bytes'] for x in removed),'estimated_reclaimed_bytes':sum(x['allocated_bytes'] for x in removed if x['links_before']==1)}
(B/'status/cleanup-receipt-20261008.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
