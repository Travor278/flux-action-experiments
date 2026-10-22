"""Headless GPU RGB acceptance probe; no policy or benchmark score."""
import os,sys
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
sys.argv.append('--allow-root')
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':128,'height':128,'renderer':'RayTracedLighting','multi_gpu':False})
try:
 import omni.replicator.core as rep
 import numpy as np,json,time,hashlib
 from pathlib import Path
 from PIL import Image
 R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
 rep.create.light(rotation=(315,0,0),intensity=3000,light_type='distant')
 cube=rep.create.cube(position=(0,0,0))
 cam=rep.create.camera(position=(300,300,300),look_at=cube)
 product=rep.create.render_product(cam,(128,128))
 rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
 for _ in range(4):rep.orchestrator.step(rt_subframes=4)
 a=np.asarray(rgb.get_data());assert a.shape==(128,128,4) and a.dtype==np.uint8
 assert float(a[:,:,:3].std())>1 and int(a[:,:,:3].max())>0
 Image.fromarray(a).save(R/'status/renderer-smoke.png')
 r={'time':time.time(),'verified':True,'shape':list(a.shape),'dtype':str(a.dtype),'rgb_std':float(a[:,:,:3].std()),'rgb_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'scope':'Headless GPU renderer and RGB extraction only, not RoboDojo task or learned policy'}
 (R/'status/renderer-smoke-verified.json').write_text(json.dumps(r,indent=2));print('RENDERER_RGB_PASS',flush=True)
finally:app.close()
