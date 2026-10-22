set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY' > "$B/status/aloha-asset-contract.json"
import json,pathlib,zipfile,yaml
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
p=B/'assets/.robotwin-sim-assets.download/embodiments.zip'
with zipfile.ZipFile(p) as z:
 names=[x for x in z.namelist() if x.endswith('/config.yml') and 'aloha' in x.lower()]
 out={}
 for n in names:
  config=yaml.safe_load(z.read(n));out[n]={k:config.get(k) for k in ['planner','urdf_path','srdf_path','arm_joints_name','gripper_name','gripper_scale','robot_pose']}
 print(json.dumps(out,indent=2))
PY
