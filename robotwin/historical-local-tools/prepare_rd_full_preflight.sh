set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import subprocess,hashlib,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');p=R/'flux-action/src/flux_action/training/checkpoint.py';backup=R/'status/checkpoint-before-gloo.py';assert not backup.exists()
old=p.read_bytes();assert old==subprocess.check_output(['git','-C',str(R/'flux-action'),'show','HEAD:src/flux_action/training/checkpoint.py']);backup.write_bytes(old)
new=(R/'scripts/checkpoint-fixed.py').read_bytes();compile(new,str(p),'exec');p.write_bytes(new)
(R/'status/full-preflight-checkpoint-support.json').write_text(json.dumps({'time':time.time(),'old_sha256':hashlib.sha256(old).hexdigest(),'new_sha256':hashlib.sha256(new).hexdigest(),'scope':'Reuse proven checkpoint planning Gloo group +CUDA sync/freecache; CPUtest/GPUrestore stillrequired inthisenvironment'},indent=2))
PY
cd "$R"
export PYTHONPATH="$R/scripts:$R/flux-action/src" OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES='' "$R/policy-venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=2 "$R/scripts/test_checkpoint_cpu.py" > "$R/status/full-preflight-checkpoint-cpu.log" 2>&1
"$R/policy-venv/bin/python" "$R/scripts/prepare_full_preflight.py" > "$R/status/full-preflight-prepare.log" 2>&1
bash -n "$R/scripts/launch_full_preflight.sh"
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
import ast,json,hashlib,time
from flux_action.training.trainer import TrainConfig
from flux_action.config import PolicyConfig
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
cfg=TrainConfig.from_file(str(R/'configs/full-preflight-v1.json'));policy=PolicyConfig(**cfg.policy);assert cfg.global_batch(2)==32 and policy.prefix_memory_spec
names=['launch_full_preflight.sh','train_full_preflight.py','cpu_power_ema.py','async_cpu_ema.py']
for n in names:
 if n.endswith('.py'):ast.parse((R/'scripts'/n).read_text())
(R/'status/full-preflight-ready.json').write_text(json.dumps({'time':time.time(),'ready':True,'files':{n:hashlib.sha256((R/'scripts'/n).read_bytes()).hexdigest() for n in names},'config_sha256':hashlib.sha256((R/'configs/full-preflight-v1.json').read_bytes()).hexdigest(),'scope':'2H100 diagnostic2updates/save/restore3/export; not formal training'},indent=2))
PY