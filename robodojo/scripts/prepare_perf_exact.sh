set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/scripts:$R/flux-action/src"
exec > "$R/status/perf-exact-prepare.log" 2>&1
"$R/policy-venv/bin/python" "$R/scripts/test_stop_outer_loop.py"
bash -n "$R/scripts/launch_profile_then_resume.sh"
bash -n "$R/scripts/launch_train_formal_stopfixed.sh"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import ast,json,time,hashlib,torch
import profile_exact_training
from train_full_preflight import recovery_fingerprint
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
source=json.loads((R/'status/formal-ready.json').read_text())
for n,h in source['files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
 x=torch.ones(8)*2
assert p.key_averages().table(sort_by='self_device_time_total',row_limit=2)
for n in ['scripts/profile_exact_training.py','scripts/launch_profile_then_resume.sh','scripts/exact_prefix_cache.py','scripts/train_full_preflight.py','scripts/train_formal_stopfixed.py','scripts/launch_train_formal_stopfixed.sh']:
 raw=(R/n).read_bytes()
 if n.endswith('.py'):ast.parse(raw)
 source['files'][n]=hashlib.sha256(raw).hexdigest()
source.update(time=time.time(),scope='Full-update exact candidate test then original production exact resume; no hyperparameter changes',ready=True)
(R/'status/perf-exact-ready.json').write_text(json.dumps(source,indent=2))
print('PERF_EXACT_PREPARED',flush=True)
PY
"$R/policy-venv/bin/python" "$R/scripts/monitor_formal_training.py"
