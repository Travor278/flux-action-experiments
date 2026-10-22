from pathlib import Path
import hashlib,json
from huggingface_hub import HfApi,snapshot_download
repo='google/paligemma-3b-pt-224';rev='35e4f46485b4d07967e7e9935bc3786aad50687c'
b=Path(__file__).resolve().parent;dst=b/'paligemma-tokenizer-authorized-cache'
try:
 info=HfApi().model_info(repo,revision=rev,files_metadata=True)
 snapshot_download(repo,revision=rev,local_dir=dst,allow_patterns=['*token*','*.model','config.json','special_tokens_map.json','preprocessor_config.json'],max_workers=4)
 expected={x.rfilename:x for x in info.siblings};files=[]
 for p in sorted(dst.rglob('*')):
  if not p.is_file() or '.cache' in p.relative_to(dst).parts:continue
  rel=p.relative_to(dst).as_posix();f=expected[rel];h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert p.stat().st_size==f.size
  if f.lfs:assert h==f.lfs.sha256
  files.append({'path':rel,'bytes':p.stat().st_size,'sha256':h})
 (dst/'verified_manifest.json').write_text(json.dumps({'repo':repo,'revision':info.sha,'files':files},indent=2),encoding='utf-8')
 print(json.dumps({'verified':True,'files':len(files),'bytes':sum(x['bytes'] for x in files)}))
except Exception as e:
 print(json.dumps({'verified':False,'error_type':type(e).__name__,'repo':repo}))
 raise SystemExit(1)
