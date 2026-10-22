import pathlib,json,time
from huggingface_hub import HfApi
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');p=B/'status/hf-final-uploads.json';r=json.loads(p.read_text());api=HfApi();out={'time':time.time(),'tasks':{}}
for task,record in r['tasks'].items():
 if not record.get('verified'):continue
 info=api.model_info(record['repo_id'],revision=record['revision'],files_metadata=True);assert info.private
 files={f.rfilename:f for f in info.siblings};assert set(files)=={'.gitattributes','model.safetensors','config.json','manifest.json'},'User requested checkpoint files only'
 for n,v in record['files'].items():
  assert files[n].size==v['bytes']
  if files[n].lfs:assert files[n].lfs.sha256==v['sha256']
 out['tasks'][task]={'verified':True,'repo_id':info.id,'revision':info.sha,'private':True,'files':{n:f.size for n,f in files.items()}}
out['complete']=len(out['tasks'])==2;(B/'status/hf-checkpoint-only-verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
