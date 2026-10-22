set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,hashlib,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
r=json.loads((R/'status/real-vae-current.json').read_text());assert 'job-REPLACE_WITH_YOUR_JOB_ID' in r['hostname'];assert r['verified'] and r['exit_code']==0
p=Path(r['run'])/'real-vae-verified.json';v=json.loads(p.read_text());assert v['verified'] and v['weights_sha256']=='db1c3211548371cf42c7230d52b5a99df5072f246d6bdfdc8fee99208ef58f01'
assert {c['task'] for c in v['cases']}=={'cover_blocks','match_and_pick_from_conveyor'}
assert all(c['causal_targets_exact'] and c['prefix_repeat_exact'] and c['valid_future_latents']==1 and c['valid_actions']==6 for c in v['cases'])
proof={'time':time.time(),'verified':True,'job_id':'job-REPLACE_WITH_YOUR_JOB_ID','result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cases':v['cases'],'scope':v['scope']}
(R/'status/real-vae-independent-audit.json').write_text(json.dumps(proof,indent=2));print('REAL_VAE_INDEPENDENT_AUDIT_PASS')
PY