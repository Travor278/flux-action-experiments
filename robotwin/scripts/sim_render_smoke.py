import json,pathlib
import numpy as np
import sapien.core as sapien
from sapien.render import set_global_config
from PIL import Image
B=pathlib.Path(__file__).resolve().parents[1]
set_global_config(max_num_materials=50000,max_num_textures=50000)
engine=sapien.Engine();renderer=sapien.SapienRenderer();engine.set_renderer(renderer)
sapien.render.set_camera_shader_dir('rt')
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser('oidn')
scene=engine.create_scene();scene.set_ambient_light([.5,.5,.5])
scene.add_point_light([1,1,2],[3,3,3])
material=renderer.create_material();material.set_base_color([.9,.1,.05,1])
builder=scene.create_actor_builder();builder.add_box_visual(half_size=[.15,.15,.15],material=material)
actor=builder.build_static(name='render_probe');actor.set_pose(sapien.Pose([0,0,.5]))
camera=scene.add_camera('probe',128,128,1.0,.01,10)
camera.set_pose(sapien.Pose([1,0,.5],[0,0,0,1]))
scene.update_render();camera.take_picture()
rgba=camera.get_picture('Color');rgb=rgba[:,:,:3]
assert rgb.shape==(128,128,3) and np.isfinite(rgb).all()
assert float(rgb.std())>.01,'Camera did not render a nonuniform scene'
Image.fromarray((np.clip(rgb,0,1)*255).astype('uint8')).save(B/'status/sim-render-probe.png')
record={'renderer':'SAPIEN rt/OIDN','shape':list(rgb.shape),'mean':float(rgb.mean()),'std':float(rgb.std()),'finite':True}
(B/'status/sim-render-check.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
