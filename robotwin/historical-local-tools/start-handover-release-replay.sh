set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,zipfile,io,h5py,numpy as np
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/handover-release-replay-v1';R.mkdir(exist_ok=True)
with zipfile.ZipFile(B/'assets/robotwin-data/dataset/handover_block/demo_clean.zip') as z:
 n=sorted(x for x in z.namelist() if x.endswith('.hdf5'))[0];seeds=[int(v) for v in z.read(next(x for x in z.namelist() if x.endswith('/seed.txt'))).split()]
 with h5py.File(io.BytesIO(z.read(n))) as h:
  keys=['left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states'];data={g:np.concatenate([h[f'{g}/{k}'][()] for k in keys],axis=1).astype('float32') for g in ['state','action']};np.savez_compressed(R/'source-episode0.npz',**data)
(R/'source.json').write_text(json.dumps({'seed':seeds[0],'episode':0,'member':n}))
PY
source "$B/scripts/sim-env.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 XDG_RUNTIME_DIR=/tmp/flux-piperx-release-replay-runtime
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
test "$free" -gt 12288
nohup flock -n "$B/status/policy-chunk-replay.lock" "$B/sim-venv/bin/python" "$B/scripts/replay_handover_release.py" > "$B/status/handover-release-replay.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/handover-release-replay.pid"
