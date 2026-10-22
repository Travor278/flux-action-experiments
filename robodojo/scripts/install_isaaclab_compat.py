"""Resolve the pinned RoboDojo/IsaacSim Starlette metadata conflict only.

The original git checkout is untouched; every runtime Python file must retain
its original SHA. This does not patch FLUX, Transformers, tasks or physics.
"""
from pathlib import Path
import shutil,hashlib,json,time,subprocess,os
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
src=R/'RoboDojo/third_party/IsaacLab/source/isaaclab';dst=R/'build/isaaclab-sim51-compat'
original=(src/'setup.py').read_text();assert original.count('"starlette==0.49.1"')==1
if not dst.exists():shutil.copytree(src,dst,ignore=shutil.ignore_patterns('*.egg-info','__pycache__','.pytest_cache'))
patched=original.replace('"starlette==0.49.1"','"starlette==0.45.3"')
(dst/'setup.py').write_text(patched)
runtime={}
for p in (src/'isaaclab').rglob('*.py'):
 q=dst/p.relative_to(src);a=hashlib.sha256(p.read_bytes()).hexdigest();assert a==hashlib.sha256(q.read_bytes()).hexdigest();runtime[str(p.relative_to(src))]=a
proof={'time':time.time(),'upstream':'afca7b09d60d8beb9c1cb28b43066499940b969b','original_setup_sha256':hashlib.sha256(original.encode()).hexdigest(),'patched_setup_sha256':hashlib.sha256(patched.encode()).hexdigest(),'change':'Installation metadata only: starlette 0.49.1 -> 0.45.3, matching RoboDojo install.sh and IsaacSim5.1 fastapi==0.115.7 requirement starlette<0.46','runtime_python_files_unchanged':runtime,'original_checkout_unchanged':True}
(R/'status/isaaclab-install-metadata-compat.json').write_text(json.dumps(proof,indent=2))
env=os.environ.copy();env['UV_CACHE_DIR']=str(R/'uv-cache')
subprocess.run(['uv','pip','install','--python',str(R/'sim-venv/bin/python'),'-e',str(dst),'-c',str(R/'configs/sim-constraints.txt')],env=env,check=True)
subprocess.run(['uv','pip','check','--python',str(R/'sim-venv/bin/python')],env=env,check=True)
from curobo.batch_motion_planner import BatchMotionPlanner,MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria
import importlib.metadata as m
(R/'status/sim-extensions-installed.json').write_text(json.dumps({'time':time.time(),'packages':{n:m.version(n) for n in ['torch','isaacsim','isaaclab','numpy','warp-lang','click','idna','huggingface-hub']},'curobo_import':True,'renderer_smoke':False,'metadata_compatibility_record':'isaaclab-install-metadata-compat.json'},indent=2))
print('ISAACLAB_METADATA_COMPAT_INSTALLED',flush=True)
