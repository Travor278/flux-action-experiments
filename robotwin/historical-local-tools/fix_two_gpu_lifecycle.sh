B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
export PYTHONPATH="$B/flux-action/src:$B/scripts" CUDA_VISIBLE_DEVICES=''
"$B/flux-action/.venv/bin/python" "$B/scripts/test_two_gpu_worker_lifecycle.py" > "$B/status/two-gpu-lifecycle-test.log" 2>&1
python3 - <<'PY'
from pathlib import Path
import shutil,json,time
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
run=S/'runs/two-gpu-feasibility-v1';save=B/'status/two-gpu-attempt2';save.mkdir(exist_ok=True)
assert run.resolve().is_relative_to(S) and not run.is_symlink() and not list(run.glob('step-*'))
for p in run.iterdir():
 assert p.is_file() and p.name in ['train.log','gpu.csv'],str(p)
 shutil.copy2(p,save/p.name)
shutil.copy2(B/'status/cpu-ema-gpu-test.log',save/'gpu-unit.log');shutil.copy2(B/'status/two-gpu-monitor.json',save/'monitor.json')
assert 'job-REPLACE_WITH_YOUR_JOB_ID' in (save/'train.log').read_text()
shutil.rmtree(run)
(B/'status/two-gpu-attempt2-cleanup.json').write_text(json.dumps({'time':time.time(),'removed':str(run),'preserved_logs':str(save),'checkpoint_existed':False}))
PY
