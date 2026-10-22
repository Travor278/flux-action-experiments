set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export UV_CACHE_DIR="$R/uv-cache" CUDA_VISIBLE_DEVICES=0
uv pip check --python "$R/policy-venv/bin/python" > "$R/status/policy-pip-check.log" 2>&1
"$R/policy-venv/bin/python" - <<'PY' > "$R/status/policy-cuda-preflight.log" 2>&1
import torch,natten,transformers,importlib.metadata as m,json,time,pathlib
R=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
assert torch.__version__.startswith('2.10.0') and torch.version.cuda=='12.8';assert transformers.__version__=='5.16.1' and m.version('lerobot')=='0.6.1'
x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16);y=x@x.T;assert torch.isfinite(y).all();torch.cuda.synchronize()
r={'time':time.time(),'verified':True,'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'lerobot':m.version('lerobot'),'natten':natten.__version__,'device':torch.cuda.get_device_name(0),'scope':'imports and CUDA BF16 matrix operation only; not full model or simulator'}
(R/'status/policy-cuda-preflight.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
PY
cd "$R/flux-action"
PYTHONPATH="$R:$R/flux-action/src" FLUX_ADAPTER_PATH="$R/XPolicyLab/policy/FLUX_RoboDojo_History/model.py" CUDA_VISIBLE_DEVICES='' "$R/policy-venv/bin/python" -m pytest tests/test_history_inference.py tests/test_history_training.py tests/test_so101_config.py "$R/scripts/test_adapter.py" -q > "$R/status/new-environment-tests.log" 2>&1