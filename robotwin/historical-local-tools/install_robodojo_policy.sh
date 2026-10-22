set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec 9> "$R/status/policy-install.lock"
flock -n 9 || exit 0
export UV_CACHE_DIR="$R/uv-cache" UV_PYTHON_INSTALL_DIR="$R/python" PYTHONNOUSERSITE=1
uv python install 3.12
uv venv --python 3.12 "$R/policy-venv"
uv pip install --python "$R/policy-venv/bin/python" torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$R/policy-venv/bin/python" -e "$R/flux-action[encoders,data,serve]" 'torch==2.10.0' 'torchvision==0.25.0' 'transformers==5.16.1' 'lerobot==0.6.1' pytest h5py
uv pip install --python "$R/policy-venv/bin/python" 'natten==0.21.6+torch2100cu128' -f https://whl.natten.org/
"$R/policy-venv/bin/python" - <<'PY'
import sys,json,time,pathlib,importlib.metadata as m
R=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
r={'time':time.time(),'python':sys.version,'packages':{n:m.version(n) for n in ['torch','transformers','lerobot','natten']}}
assert sys.version_info[:2]==(3,12)
assert r['packages']['transformers']=='5.16.1' and r['packages']['lerobot']=='0.6.1'
assert r['packages']['torch'].startswith('2.10.0')
(R/'status/policy-environment-verified.json').write_text(json.dumps(r,indent=2))
PY
