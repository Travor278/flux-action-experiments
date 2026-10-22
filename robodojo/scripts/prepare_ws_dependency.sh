set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/sim-venv/bin/python" - <<'PY'
from pathlib import Path
import msgpack_numpy,hashlib,json,time,shutil
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
src=Path(msgpack_numpy.__file__);dst=R/'ws-deps/msgpack_numpy.py'
dst.parent.mkdir(exist_ok=True)
if dst.exists():assert dst.read_bytes()==src.read_bytes()
else:shutil.copyfile(src,dst)
(R/'status/ws-codec-dependency.json').write_text(json.dumps({'time':time.time(),'source':str(src),'destination':str(dst),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'scope':'Isolated pure Python codec copied from already installed SIM environment; no active training environment changes'},indent=2))
PY
