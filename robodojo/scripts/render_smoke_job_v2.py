"""Instrumented native-resolution camera probe; no policy/benchmark score."""
import os,sys,time,json,faulthandler
from pathlib import Path
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
out=Path(os.environ['FLUX_RENDER_STATUS_DIR']);out.mkdir(parents=True,exist_ok=True)
def phase(name,**extra):
 r={'time':time.time(),'pid':os.getpid(),'phase':name,**extra}
 with (out/'phases.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 print(json.dumps(r),flush=True)
faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=True)
os.environ['OMNI_KIT_ACCEPT_EULA']='YES';sys.argv.append('--allow-root')
phase('app_loading')
from isaaclab.app import AppLauncher
launcher=AppLauncher({'headless':True,'enable_cameras':True,'device':'cuda:0','experience':str(R/'RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.rendering.kit'),'kit_args':'--/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true'})
app=launcher.app;phase('app_ready')
try:
 import omni.replicator.core as rep
 import omni.timeline
 import numpy as np,hashlib
 from PIL import Image
 rep.create.light(rotation=(315,0,0),intensity=3000,light_type='distant')
 cube=rep.create.cube(position=(0,0,0));cam=rep.create.camera(position=(300,300,300),look_at=cube)
 product=rep.create.render_product(cam,(640,480));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
 rep.orchestrator.set_capture_on_play(True);omni.timeline.get_timeline_interface().play();phase('camera_attached')
 valid=0
 for tick in range(300):
  app.update();a=np.asarray(rgb.get_data())
  if tick%20==0:phase('camera_updates',tick=tick,shape=list(a.shape))
  if a.shape==(480,640,4) and a.dtype==np.uint8 and float(a[:,:,:3].std())>1:valid+=1
  else:valid=0
  if valid>=3:break
 assert valid>=3,'No three consecutive nonempty RGB frames'
 Image.fromarray(a).save(out/'renderer-smoke.png')
 r={'time':time.time(),'verified':True,'shape':list(a.shape),'dtype':str(a.dtype),'rgb_std':float(a[:,:,:3].std()),'rgb_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'valid_frames':valid,'scope':'Actual headless camera rendering only, not policy success'}
 (out/'renderer-smoke-verified.json').write_text(json.dumps(r,indent=2));phase('rgb_verified')
finally:
 phase('app_closing');app.close();phase('app_closed');faulthandler.cancel_dump_traceback_later()
