set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import hashlib,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');target=R/'flux-action/src/flux_action/processing/temporal_masking.py';backup=R/'status/temporal-mask-before-real-vae-fix.py'
assert not backup.exists();old=target.read_bytes();backup.write_bytes(old)
new=(R/'scripts/temporal_masking.py').read_bytes();compile(new,str(target),'exec');temp=target.with_suffix('.tmp');temp.write_bytes(new);temp.replace(target)
r={'time':time.time(),'old_sha256':hashlib.sha256(old).hexdigest(),'new_sha256':hashlib.sha256(new).hexdigest(),'reason':'TemporalMerging duplicates first input only for odd length;32framefutureclip endpoints3,7,...31 not0,4,...28. RealVAEprobe a864 failed first causalcheck, logs preserved. No formaltraining used this experimentalRoboDojo mask.'}
(R/'status/temporal-mask-real-vae-repair.json').write_text(json.dumps(r,indent=2))
PY
cd "$R/scripts"
OMP_NUM_THREADS=2 "$R/policy-venv/bin/python" -m pytest test_temporal_masks.py test_temporal_policy_integration.py test_prefix_policy.py "$R/flux-action/tests/test_history_training.py" "$R/flux-action/tests/test_history_inference.py" -q > "$R/status/real-vae-mask-repair-tests.log" 2>&1
printf PASS > "$R/status/real-vae-mask-repair-tests-passed.txt"