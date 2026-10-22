set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
bash -n "$R/scripts/launch_real_vae_probe.sh"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import ast,json,hashlib,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
ast.parse((R/'scripts/probe_real_vae.py').read_text())
assert (R.parent/'flux_piperx_20261008/assets/flux-base/video_vae.safetensors').is_file()
assert (R/'status/prefix-policy-tests-passed.txt').read_text().strip()=='PASS'
assert json.loads((R/'status/all-video-decode-independent-audit.json').read_text())['verified']
from flux_action.models.video_vae import load_video_vae
from flux_action.processing.prefix_memory import encode_prefix
j={'time':time.time(),'ready':True,'files':{n:hashlib.sha256((R/'scripts'/n).read_bytes()).hexdigest() for n in ['probe_real_vae.py','launch_real_vae_probe.sh']},'scope':'Single4090 realVAE readonly probe; not formaltraining'}
(R/'status/real-vae-job-ready.json').write_text(json.dumps(j,indent=2));print('REAL_VAE_JOB_READY')
PY