from pathlib import Path
import json,hashlib,tarfile,io,time,subprocess
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=Path(str(B).replace('/hdd/','/ssd/'))
groups={k:{} for k in ['01-results','02-configs','03-source','04-data-provenance','05-runtime-evidence','06-publication-cleanup']}
def add(group,p,root):
 if p.is_symlink() or not p.is_file() or p.stat().st_size>8_000_000:return
 if p.suffix not in {'.json','.jsonl','.py','.sh','.toml','.md','.txt','.patch','.yml','.yaml','.log'}:return
 if any(x in p.parts for x in ['.git','.venv','__pycache__','.cache','text_encoder']):return
 try:text=p.read_text()
 except (UnicodeError,OSError):return
 groups[group][str(p.relative_to(root))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':text}
for p in (B/'evaluation').rglob('*'):add('01-results',p,B)
for p in (B/'configs').rglob('*'):add('02-configs',p,B)
for folder in ['scripts','flux-action-tail-v1/src','flux-action-tail-v1/tests']:
 for p in (B/folder).rglob('*'):add('03-source',p,B)
for folder in ['assets/index','assets/index-tail-v1','assets/pi05-v3']:
 for p in (B/folder).rglob('*.json'):add('04-data-provenance',p,B)
for p in (B/'assets').glob('*.json'):add('04-data-provenance',p,B)
for root in [B/'status',B/'runs',S/'runs']:
 for p in root.rglob('*'):
  add('06-publication-cleanup' if ('hf-' in p.name or 'cleanup' in p.name) else '05-runtime-evidence',p, B if root!=S/'runs' else S)
add('05-runtime-evidence',B/'HANDOFF.md',B)
archive=B/'status/robotwin-records-20261013.tar.gz'
manifest={'time':time.time(),'archive':str(archive),'records':{},'payloads_excluded':['weights','optimizer','dataset arrays','videos']}
with tarfile.open(archive,'w:gz') as tar:
 for name,items in groups.items():
  data=json.dumps({'record':name,'files':items},ensure_ascii=False).encode();info=tarfile.TarInfo(name+'.json');info.size=len(data);tar.addfile(info,io.BytesIO(data));manifest['records'][name]={'files':len(items),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
manifest['bytes']=archive.stat().st_size;manifest['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
(B/'status/robotwin-records-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
