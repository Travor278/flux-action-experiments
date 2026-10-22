set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
mkdir -p "$R/status" "$R/scripts" "$R/configs"
export GIT_LFS_SKIP_SMUDGE=1
if [ ! -d "$R/RoboDojo/.git" ]; then git clone --depth 1 https://github.com/RoboDojo-Benchmark/RoboDojo.git "$R/RoboDojo"; fi
git -C "$R/RoboDojo" checkout 09bd5f5e762b949f5858ece31347ce62057b3a5e
if [ ! -d "$R/XPolicyLab/.git" ]; then git clone --depth 1 https://github.com/XPolicyLab/XPolicyLab.git "$R/XPolicyLab"; fi
git -C "$R/XPolicyLab" checkout f6e309e8ece20bfa33c0e5d7b05a5f3287522757
if [ ! -d "$R/flux-action/.git" ]; then git clone --depth 1 https://github.com/black-forest-labs/flux-action.git "$R/flux-action"; fi
git -C "$R/flux-action" checkout 830f1ce178d7aea52f18fde31d8e8e36ce6bae21
"$B/flux-action/.venv/bin/python" - <<'PY'
import json,time,pathlib,sys,subprocess,importlib.metadata as m
from huggingface_hub import HfApi
R=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');api=HfApi()
r={'time':time.time(),'tasks':['cover_blocks','match_and_pick_from_conveyor'],'training_episodes_per_task':100,'test_episodes_per_task':50,'phase':'source_and_data_inventory','training_started':False,'revisions':{},'runtime':{'python':sys.version}}
for name in ['RoboDojo','XPolicyLab','flux-action']:r['revisions'][name]=subprocess.check_output(['git','-C',str(R/name),'rev-parse','HEAD'],text=True).strip()
for name in ['torch','transformers','lerobot','huggingface_hub','hf_xet']:
 try:r['runtime'][name]=m.version(name)
 except m.PackageNotFoundError:r['runtime'][name]=None
(R/'status/preparation.json').write_text(json.dumps(r,indent=2))
info=api.dataset_info('RoboDojo-Benchmark/RoboDojo');r['dataset_revision']=info.sha
selected=[]
for f in api.list_repo_tree(info.id,repo_type='dataset',revision=info.sha,recursive=True):
 if not hasattr(f,'size'):continue
 if f.path.startswith('Assets/') or (f.path.startswith('data/') and any(t in f.path for t in r['tasks'])):
  selected.append({'path':f.path,'bytes':f.size,'lfs_sha256':f.lfs.sha256 if f.lfs else None})
r['remote_files']=selected;r['total_candidate_bytes']=sum(f['bytes'] for f in selected)
(R/'status/preparation.json').write_text(json.dumps(r,indent=2));print('SOURCE_AND_DATA_INVENTORY_COMPLETE',flush=True)
PY
