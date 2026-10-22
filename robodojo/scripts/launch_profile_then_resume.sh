#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
S=${R/\/hdd\//\/ssd\/}
TASK=match_and_pick_from_conveyor
OUT="$S/runs/$TASK-10k-prefix-v1"
export PATH="$R/policy-venv/bin:$PATH" PYTHONPATH="$R/scripts:$R/flux-action/src" PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PERF_ORIGINAL_CONFIG="$R/configs/$TASK-10k-prefix-v1.json"
RUN="$S/runs/perf-exact-20261014"
export PERF_RUN="$RUN"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/launch.log") 2>&1
python "$R/scripts/check_formal_gate.py" "$TASK"
exec 8>"$R/status/perf-exact.lock"
flock -n 8 || exit 3
python - <<'PY'
from pathlib import Path
import json,os,time,pickle,torch,socket
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
run=Path(os.environ['PERF_RUN']);c=json.loads(Path(os.environ['PERF_ORIGINAL_CONFIG']).read_text());out=Path(c['output_dir'])
stop=json.loads((out/'STOPPED_SAFELY.json').read_text());p=max(out.glob('step-*/COMPLETE'),key=lambda x:int(x.parent.name.split('-')[1])).parent;step=int(p.name.split('-')[1]);assert step>=stop['step']
assert (p/'COMPLETE').is_file()
state=json.loads((p/'state.json').read_text());assert state['step']==step and state['world_size']==2
components={}
for name in ('model','optimizer','ema_0p10','ema_0p05'):
 with (p/name/'.metadata').open('rb') as f:m=pickle.load(f)
 components[name]=len(m.state_dict_metadata)
 for v in m.storage_data.values():
  q=(p/name/v.relative_path).resolve();assert q.is_relative_to((p/name).resolve()) and v.offset+v.length<=q.stat().st_size
assert components=={'model':451,'optimizer':1367,'ema_0p10':447,'ema_0p05':447}
for rank in (0,1):assert isinstance(torch.load(p/f'rng/rank-{rank:04d}.pt',weights_only=True),dict)
for mode in ('baseline','cached'):
 d={**c,'output_dir':str(run/mode),'resume':str(p),'steps':step+8,'checkpoint_every':1000000,'log_every':1,'keep_checkpoints':None}
 target=run/f'{mode}.json'
 if target.exists():assert json.loads(target.read_text())==d
 else:target.write_text(json.dumps(d,indent=2))
record={'time':time.time(),'hostname':socket.gethostname(),'phase':'checkpoint_audited','source_checkpoint':str(p),'step':step,'components':components,'run':str(run)}
(R/'status/perf-exact-current.json').write_text(json.dumps(record,indent=2))
PY
code=0
for mode in baseline cached; do
 if [ -f "$RUN/$mode/comparison-rank0.json" ] && [ -f "$RUN/$mode/comparison-rank1.json" ]; then continue; fi
 nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > "$RUN/$mode-gpu.csv" &
 mon=$!
 set +e
 timeout --signal=TERM --kill-after=30s 1500s python -m torch.distributed.run --standalone --nproc_per_node=2 "$R/scripts/profile_exact_training.py" "$RUN/$mode.json" "$mode" > "$RUN/$mode.log" 2>&1
 code=$?
 set -e
 kill "$mon" 2>/dev/null || true
 if [ "$code" -ne 0 ]; then break; fi
done
export PERF_EXIT_CODE="$code"
python - <<'PY'
from pathlib import Path
import json,os,time,statistics
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');run=Path(os.environ['PERF_RUN'])
r=json.loads((R/'status/perf-exact-current.json').read_text());r.update(finished=time.time(),exit_code=int(os.environ['PERF_EXIT_CODE']),verified=False)
if r['exit_code']==0:
 comparisons=[]
 for rank in (0,1):
  a=json.loads((run/f'baseline/comparison-rank{rank}.json').read_text());b=json.loads((run/f'cached/comparison-rank{rank}.json').read_text())
  before=a['before']==b['before'];after=a['after']==b['after']
  seconds=lambda x:statistics.median(v['metrics']['step_time'] for v in x['updates'][2:] if not v['profiled'])
  comparisons.append({'rank':rank,'before_exact':before,'after_exact':after,'changed_components':[k for k in a['after'] if a['after'][k]!=b['after'][k]],'baseline_seconds':seconds(a),'cached_seconds':seconds(b)})
 r.update(comparisons=comparisons,verified=all(x['before_exact'] and x['after_exact'] for x in comparisons))
r['phase']='comparison_complete';r['production_optimization_adopted']=False
(run/'result.json').write_text(json.dumps(r,indent=2));(R/'status/perf-exact-current.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r),flush=True)
PY
# Restore the untouched production trajectory regardless of diagnostic result.
# The diagnostic updates never replace the source checkpoint or its RNG.
python - <<'PY'
from pathlib import Path
import json,time,os
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');run=Path(os.environ['PERF_RUN'])
out=Path(json.loads(Path(os.environ['PERF_ORIGINAL_CONFIG']).read_text())['output_dir'])
r=json.loads((R/'status/perf-exact-current.json').read_text());assert (Path(r['source_checkpoint'])/'COMPLETE').exists()
archive=run/'production-stop-evidence';archive.mkdir(exist_ok=True)
for name in ['STOP_AFTER_UPDATE','STOPPED_SAFELY.json']:
 p=out/name
 if p.exists():
  assert not (archive/name).exists();p.rename(archive/name)
r['production_resume_requested']=time.time();(R/'status/perf-exact-current.json').write_text(json.dumps(r,indent=2))
PY
flock -u 8
exec bash "$R/scripts/launch_train_formal_stopfixed.sh" "$TASK"
