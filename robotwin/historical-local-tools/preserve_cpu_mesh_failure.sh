B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cp "$B/status/cpu-ema-gpu-test.log" "$B/status/cpu-ema-gpu-test-attempt1.log"
cp "$B/status/two-gpu-monitor.json" "$B/status/two-gpu-failure-attempt1.json"
"$B/flux-action/.venv/bin/python" - <<'PY' > "$B/status/cpu-mesh-api.txt"
import inspect
from torch.distributed.device_mesh import DeviceMesh
print(inspect.signature(DeviceMesh.from_group))
PY
