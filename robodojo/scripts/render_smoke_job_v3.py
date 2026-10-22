"""Explicit USD scene + native RGB probe, independent of Replicator graph authoring."""
import os,sys,time,json,faulthandler,traceback
from pathlib import Path
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
out=Path(os.environ['FLUX_RENDER_STATUS_DIR']);out.mkdir(parents=True,exist_ok=True)
def phase(name,**extra):
 r={'time':time.time(),'pid':os.getpid(),'phase':name,**extra}
 with (out/'phases.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 print(json.dumps(r),flush=True)
faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=True)
os.environ['OMNI_KIT_ACCEPT_EULA']='YES';sys.argv.append('--allow-root');phase('app_loading')
from isaaclab.app import AppLauncher
launcher=AppLauncher({'headless':True,'enable_cameras':True,'device':'cuda:0','experience':str(R/'RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.rendering.kit'),'kit_args':'--/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true'})
app=launcher.app;phase('app_ready')
try:
 import omni.replicator.core as rep
 import omni.timeline,omni.usd
 from pxr import UsdGeom,UsdLux,Gf
 import numpy as np,hashlib
 from PIL import Image
 stage=omni.usd.get_context().get_stage()
 UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.0)
 cube=UsdGeom.Cube.Define(stage,'/World/ProbeCube');cube.CreateSizeAttr(1.0);cube.CreateDisplayColorAttr([Gf.Vec3f(.8,.08,.02)])
 light=UsdLux.DistantLight.Define(stage,'/World/ProbeLight');light.CreateIntensityAttr(3000.0)
 UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(315,0,0))
 camera=UsdGeom.Camera.Define(stage,'/World/ProbeCamera');camera.CreateClippingRangeAttr(Gf.Vec2f(.01,1000.0));camera.CreateFocalLengthAttr(24.0)
 pose=Gf.Matrix4d().SetLookAt(Gf.Vec3d(3,3,3),Gf.Vec3d(0,0,0),Gf.Vec3d(0,0,1)).GetInverse()
 UsdGeom.Xformable(camera).AddTransformOp().Set(pose)
 assert stage.GetPrimAtPath('/World/ProbeCube').IsValid() and stage.GetPrimAtPath('/World/ProbeCamera').IsValid()
 product=rep.create.render_product('/World/ProbeCamera',(640,480));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
 stage.GetRootLayer().Export(str(out/'probe-scene.usda'))
 rep.orchestrator.set_capture_on_play(True);omni.timeline.get_timeline_interface().play();phase('camera_attached',camera_pose=str(pose))
 valid=0;a=np.empty((0,),np.uint8)
 for tick in range(300):
  app.update();a=np.asarray(rgb.get_data())
  stats={'shape':list(a.shape),'dtype':str(a.dtype)}
  if a.ndim==3:stats.update(rgb_std=float(a[:,:,:3].std()),minimum=int(a[:,:,:3].min()),maximum=int(a[:,:,:3].max()))
  if tick%20==0:phase('camera_updates',tick=tick,**stats)
  if a.shape==(480,640,4) and a.dtype==np.uint8 and float(a[:,:,:3].std())>1:valid+=1
  else:valid=0
  if valid>=3:break
 if a.ndim==3 and a.dtype==np.uint8:Image.fromarray(a).save(out/'last-frame.png')
 assert valid>=3,'No three consecutive nonuniform RGB frames'
 r={'time':time.time(),'verified':True,'shape':list(a.shape),'dtype':str(a.dtype),'rgb_std':float(a[:,:,:3].std()),'rgb_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'valid_frames':valid,'scope':'GPU-rendered explicit USD cube, not robot task or policy success'}
 (out/'renderer-smoke-verified.json').write_text(json.dumps(r,indent=2));phase('rgb_verified')
except Exception as error:
 (out/'probe-failure.json').write_text(json.dumps({'time':time.time(),'type':type(error).__name__,'message':str(error)},indent=2));traceback.print_exc();phase('probe_failed',error=type(error).__name__)
 raise
finally:
 phase('app_closing');app.close();faulthandler.cancel_dump_traceback_later()
