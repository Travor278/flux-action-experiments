set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
bash -n "$B/scripts/prepare_data.sh"
nohup bash "$B/scripts/prepare_data.sh" > "$B/status/prepare-final.log" 2>&1 < /dev/null &
echo $! > "$B/status/prepare-final.pid"
"$B/flux-action/.venv/bin/python" - <<'PY' > "$B/status/cuda-runtime-preflight.json"
import torch,json
assert torch.cuda.is_available()
x=torch.randn((128,128),device='cuda',dtype=torch.bfloat16)
y=x@x
assert torch.isfinite(y).all()
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'bf16_matmul':'finite'}))
PY
