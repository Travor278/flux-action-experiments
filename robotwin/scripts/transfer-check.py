import json,pathlib,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
def sample():
 out={'time':time.time(),'files':[]}
 for p in (B/'assets').rglob('*'):
  if p.is_file() and p.stat().st_size>1000000: out['files'].append({'path':str(p.relative_to(B)),'size':p.stat().st_size,'blocks':p.stat().st_blocks*512})
 out['network']=pathlib.Path('/proc/net/dev').read_text()
 return out
rows=[sample()]; time.sleep(15);rows.append(sample())
(B/'status/transfer-samples.json').write_text(json.dumps(rows,indent=2))
