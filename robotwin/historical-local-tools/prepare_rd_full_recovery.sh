set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/scripts"
OMP_NUM_THREADS=2 "$R/policy-venv/bin/python" -m unittest test_recovery_fingerprint > "$R/status/recovery-fingerprint-tests.log" 2>&1
bash -n "$R/scripts/launch_full_preflight_recover.sh"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,hashlib,time,ast
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
assert json.loads((R/'status/full-preflight-step2-independent-audit.json').read_text())['verified']
ast.parse((R/'scripts/train_full_preflight.py').read_text())
p=R/'status/full-preflight-recovery-ready.json'
p.write_text(json.dumps({'time':time.time(),'ready':True,'files':{n:hashlib.sha256((R/'scripts'/n).read_bytes()).hexdigest() for n in ['train_full_preflight.py','launch_full_preflight_recover.sh','cpu_power_ema.py','async_cpu_ema.py']},'config_sha256':hashlib.sha256((R/'configs/full-preflight-v1.json').read_bytes()).hexdigest(),'scope':'Preserve step2, resume2->save3 and fingerprints; independentrestore3->4 verifiesmodeloptimizerEMA/RNG/data fingerprints; no originalstep2presavehashclaim'},indent=2))
PY