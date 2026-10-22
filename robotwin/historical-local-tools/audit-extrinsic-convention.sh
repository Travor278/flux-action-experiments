set -euo pipefail
"/workspace/hdd/codex_remote_ops/flux_piperx_20261008/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time,numpy as np
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/expert-replay-click';out={'time':time.time(),'cases':[]};flip=np.diag([1.,-1.,-1.])
for f in R.glob('episode-*/result.json'):
 r=json.loads(f.read_text())
 for cam,c in r['camera_errors'].items():
  if 'runtime_extrinsic' not in c:continue
  a=np.array(c['source_extrinsic']);b=np.array(c['runtime_extrinsic']);out['cases'].append({'episode':r['episode'],'cam':cam,'rotation_transpose_then_cv_gl_flip_error':float(np.max(np.abs(a[:,:3]-flip@b[:,:3].T))),'translation_direct_error':float(np.max(np.abs(a[:,3]-b[:,3]))),'source_extrinsic':a.tolist(),'runtime_extrinsic':b.tolist()})
(B/'status/expert-replay-extrinsic-convention.json').write_text(json.dumps(out,indent=2))
PY
