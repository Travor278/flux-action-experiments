"""Real WebSocket diagnostic export integration; no benchmark score."""
import os,sys,json,time,hashlib,faulthandler,importlib,traceback
from pathlib import Path
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
task=sys.argv[1];assert task in ('cover_blocks','match_and_pick_from_conveyor')
out=Path(os.environ['FLUX_SCENE_RUN'])/task;out.mkdir(parents=True,exist_ok=True)
def phase(name,**kw):
    record={'time':time.time(),'phase':name,**kw}
    with (out/'phases.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
faulthandler.enable()
sys.argv=[sys.argv[0],'--allow-root']
os.environ['ROBODOJO_RUN_ID']='control-smoke-'+out.parent.name+'-'+task
from isaaclab.app import AppLauncher
phase('app_loading')
launcher=AppLauncher({'headless':True,'enable_cameras':True,'device':'cuda:0','experience':str(R/'RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.rendering.kit'),'kit_args':'--/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true'})
app=launcher.app;phase('app_ready')
try:
    # The minimal rendering experience enables RGB annotators but not the
    # standalone Camera extension used by RoboDojo's actual camera manager.
    import omni.kit.app
    manager=omni.kit.app.get_app().get_extension_manager()
    extensions=['isaacsim.core.cloner','isaacsim.core.api','isaacsim.core.prims','isaacsim.core.utils',
                'isaacsim.core.simulation_manager','isaacsim.sensors.camera',
                'isaacsim.replicator.behavior','isaacsim.asset.gen.conveyor']
    for extension in extensions:
        manager.set_extension_enabled_immediate(extension,True)
        assert manager.is_extension_enabled(extension),extension
    from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
    from isaacsim.sensors.camera import Camera
    phase('camera_extension_ready',camera_module=Camera.__module__)
    phase('imports_begin')
    import numpy as np,torch
    from PIL import Image
    from omegaconf import OmegaConf
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config,process_randomization
    from env.global_configs import ENV_CONFIG_PATH
    phase('base_imports_ready')
    import src.eval_client.eval_env as module
    phase('eval_import_ready')
    from conveyor_graph_compat import install_conveyor_graph_compat
    install_conveyor_graph_compat(out/'graph-compat.jsonl')
    base=Path(ENV_CONFIG_PATH);ec=load_yaml(str(base/'arx_x5.yml'))
    ec.update(task_name=task,num_envs=1,device_id=0,eval_batch=False,policy_name='FLUX_RoboDojo_History',additional_info='real_ws_smoke_not_score',seed=0,physx_monitor_enabled=False)
    cfg=OmegaConf.create({kind:load_yaml(str(base/kind/(ec['config'][kind]+'.yml'))) for kind in ('sim','scene','camera','robot')})
    cfg.task_env=load_yaml(str(R/'RoboDojo/task/RoboDojo/config'/f'{task}.yml'))
    cfg.eval_cfg=ec;cfg.deploy_cfg={'policy_name':'FLUX_RoboDojo_History','port':int(os.environ['FLUX_WS_PORT']),'host':'127.0.0.1'}
    cfg=process_randomization(cfg);cfg,_=process_config(cfg,task)
    # Match the official entry point: CPU physics/state by default, GPU RTX
    # rendering selected through AppLauncher independently.
    cfg.sim.scene.num_envs=1;cfg.sim.seed=[0]
    assert cfg.sim.get('device','cpu')=='cpu'
    cfg.camera.default_frequency=25
    phase('environment_constructing')
    env=module.create_eval_env(cfg,app)
    seeds=env.seed_manager.get_seeds(max_count=1);assert seeds==[0]
    phase('layout_reset',layout_ids=seeds)
    env.reset(seed=seeds);phase('layout_ready')
    from conveyor_graph_compat import restore_loaded_conveyor_graph
    if task=='match_and_pick_from_conveyor':restore_loaded_conveyor_graph(omni.usd.get_context().get_stage(),out/'graph-compat.jsonl')
    # Official EvalEnv.run_eval registers task checks before the policy loop.
    # Without registration get_reward(empty checks) returns1 and ends at tick1.
    env.run_reward()
    if hasattr(env,'get_score'):env.get_score()
    assert any(env.reward_manager.check_list[0])
    phase('reward_registered',check_count=len(env.reward_manager.check_list[0]))
    adapter=importlib.import_module('XPolicyLab.policy.FLUX_RoboDojo_History.model')
    from XPolicyLab.utils.process_data import pack_robot_state,get_robot_action_dim_info
    info=get_robot_action_dim_info('arx_x5')
    receipts=[]
    for tick in range(53):
        assert not env.is_episode_end(), 'Diagnostic episode ended before interface coverage; not a benchmark result'
        raw=env.get_obs()
        state=np.asarray(pack_robot_state(raw,'joint',info),dtype=np.float32)
        assert state.shape==(14,) and np.isfinite(state).all()
        images={aliases[0]:np.ascontiguousarray(adapter.image_chw(raw,aliases).transpose(1,2,0)) for aliases in adapter.CAMERAS.values()}
        assert all(im.shape==(480,640,3) and im.dtype==np.uint8 and float(im.std())>1 for im in images.values())
        obs={'state':state,'images':images,'instruction':raw['instruction']}
        env.model_client.call(func_name='update_obs',obs=obs)
        actions=env.model_client.call(func_name='get_action')
        assert len(actions)==1
        flat=np.asarray(pack_robot_state({'state':actions[0]},'joint',info),dtype=np.float32)
        assert flat.shape==(14,) and np.isfinite(flat).all()
        env.take_action(actions[0])
        assert env.take_action_cnt==[tick+1]
        receipts.append({'tick':tick,'action_sha256':hashlib.sha256(flat.tobytes()).hexdigest(),'state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),'rgb_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in images.items()}})
        phase('real_ws_control',tick=tick,executed=env.take_action_cnt[0])
    env.model_client.call(func_name='reset')
    record={'time':time.time(),'verified':True,'task':task,'control_ticks':53,'layout_id':0,'receipts':receipts,'scope':'Real diagnostic export4 policy over official WebSocket, 53 official control ticks, no benchmark score or trained policy claim'}
    (out/'ws-control-verified.json').write_text(json.dumps(record,indent=2))
    env.close();phase('complete')
except BaseException as exc:
    # SimulationApp.close may terminate with zero before Python prints an
    # unwinding exception. Persist and flush the cause before invoking it.
    error={'time':time.time(),'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    (out/'failure.json').write_text(json.dumps(error,indent=2))
    print(error['traceback'],file=sys.stderr,flush=True)
    phase('failed',error_type=error['type'],error_message=error['message'])
    # Kit shutdown can itself wait indefinitely after a failed reset. The
    # process owns no other experiment; durable cause is saved above.
    os._exit(1)
finally:
    app.close()
