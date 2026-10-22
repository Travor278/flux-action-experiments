set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import json,time,hashlib,shutil,subprocess
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013').resolve();target=R/'runs/checkpoint-cpu-smoke'
assert target.resolve()==target and target.parent==R/'runs' and target.name=='checkpoint-cpu-smoke'
assert 'CPU_2RANK_MODEL_OPTIMIZER_DUAL_EMA_SAVE_RESTORE_PASS' in (R/'status/full-preflight-checkpoint-cpu.log').read_text()
assert (target/'step-1/COMPLETE').exists()
ps=subprocess.check_output(['ps','-ww','-eo','pid,args'],text=True)
assert not any(str(target) in l or str(R/'scripts/test_checkpoint_cpu.py') in l for l in ps.splitlines())
files=[p for p in target.rglob('*') if p.is_file()];size=sum(p.stat().st_size for p in files)
evidence=R/'status/checkpoint-cpu-smoke-cleanup-evidence';evidence.mkdir(exist_ok=True)
for name in ['state.json','config.json','COMPLETE']:shutil.copy2(target/'step-1'/name,evidence/name)
manifest=[{'path':str(p.relative_to(target)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(evidence/'manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.rmtree(target)
record={'time':time.time(),'deleted':str(target),'freed_bytes':size,'reason':'Completed standalone CPU checkpoint regression; retained proof/log/source and compact metadata','active_ssd_training_checkpoint_untouched':True}
(R/'status/checkpoint-cpu-smoke-cleanup.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
PY