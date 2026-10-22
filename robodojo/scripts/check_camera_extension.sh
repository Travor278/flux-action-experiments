R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/scene-camera-extension-inventory.json"
from pathlib import Path
import json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
run=Path(json.loads((R/'status/scene-control-current.json').read_text())['run'])
for n in ['probe_scene_control.py','launch_scene_control.sh']:(run/('original-'+n)).write_bytes((R/'scripts'/n).read_bytes())
root=R/'sim-venv/lib/python3.11/site-packages/isaacsim'
paths=[str(p) for p in root.rglob('extension.toml') if 'sensors.camera' in str(p)]
kit=R/'RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.rendering.kit'
print(json.dumps({'time':time.time(),'camera_extensions':paths,'kit':kit.read_text()},indent=2))
PY
