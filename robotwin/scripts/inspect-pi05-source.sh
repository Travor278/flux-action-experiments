set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
P=/workspace/hdd/codex_remote_ops/A2R_pi05_20261007/venv
"$P/bin/python" - <<'PY'
import pathlib,json,inspect
import lerobot,transformers
from transformers.models.siglip import check
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');P=B.parent/'A2R_pi05_20261007/assets'
patches=list(pathlib.Path(lerobot.__file__).parent.rglob('transformers_replace'))
meta={}
for name in ['pi05-base-b211f3d4','paligemma-tokenizer']:
 root=P/name
 meta[name]={'files':[x.name for x in root.iterdir()],'hf_commit_metadata':{str(x.relative_to(root)):x.read_text().splitlines()[:2] for x in root.glob('.cache/huggingface/download/*.metadata')}}
(B/'status/pi05-source-details.json').write_text(json.dumps({'transformers_path':transformers.__file__,'check_file':check.__file__,'check_source':inspect.getsource(check),'patch_sources':[str(x) for x in patches],'assets':meta},indent=2))
PY
