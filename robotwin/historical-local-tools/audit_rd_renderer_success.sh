set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np,json,hashlib,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');r=json.loads((R/'status/renderer-job-current.json').read_text())
assert 'job-REPLACE_WITH_YOUR_JOB_ID' in r['hostname'] and r['render_passed'] and r['exit_code']==0
p=Path(r['run'])/'last-frame.png';a=np.asarray(Image.open(p).convert('RGBA'));assert a.shape==(480,640,4)
assert hashlib.sha256(a.tobytes()).hexdigest()==r['render']['rgb_sha256']
x=a[:,:,:3].astype(np.int16);red=(x[:,:,0]>x[:,:,1]+15)&(x[:,:,0]>x[:,:,2]+15);assert int(red.sum())>100
out={'time':time.time(),'verified':True,'job_id':'job-REPLACE_WITH_YOUR_JOB_ID','driver':r['driver'],'run':r['run'],'png_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_rgba_sha256':r['render']['rgb_sha256'],'red_dominant_pixels':int(red.sum()),'shape':list(a.shape),'scope':'independentPNGdecode/rawSHA and red-cube RGB channel-order audit, not benchmarkscore'}
(R/'status/renderer-compatible-node-verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
PY