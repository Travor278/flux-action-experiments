from pathlib import Path
import json,time,os
os.environ['HF_HUB_DISABLE_XET']='1'
import pyarrow.parquet as pq
from huggingface_hub import HfApi,hf_hub_download,set_client_factory
import httpx
set_client_factory(lambda:httpx.Client(timeout=httpx.Timeout(60,connect=15),follow_redirects=True))
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');api=HfApi();repo='RoboDojo-Benchmark/RoboDojo';prefix='data/RoboDojo_lerobot_v30_video'
revision=api.dataset_info(repo).sha
files=list(api.list_repo_tree(repo,path_in_repo=prefix+'/meta',revision=revision,repo_type='dataset',recursive=True))
paths=[]
for f in files:
 if not hasattr(f,'size'):continue
 print('METADATA '+f.path,flush=True)
 paths.append(Path(hf_hub_download(repo,f.path,repo_type='dataset',revision=revision,local_dir=R/'official-source')))
info=json.loads((R/'official-source'/prefix/'meta/info.json').read_text());assert info['codebase_version']=='v3.0' and info['fps']==25
tasks=pq.read_table(R/'official-source'/prefix/'meta/tasks.parquet').to_pylist()
eps=[]
for p in paths:
 if '/episodes/' in str(p) and p.suffix=='.parquet':eps+=pq.read_table(p).to_pylist()
selected={}
for task,keyword in [('cover_blocks','cover'),('match_and_pick_from_conveyor','conveyor')]:
 rows=[e for e in eps if keyword in str(e.get('tasks','')).lower()]
 selected[task]={'count':len(rows),'rows':rows}
r={'time':time.time(),'revision':revision,'prefix':prefix,'info':info,'tasks':tasks,'episode_columns':list(eps[0]),'selected':selected}
(R/'status/official-v3-inspection.json').write_text(json.dumps(r,indent=2,default=str));print('V3_METADATA_READ',flush=True)
