"""Fixed-layout official final50 evaluation; infrastructure failures stop without replacing scenes."""
import os,sys,json,time,hashlib,faulthandler,importlib,traceback
from pathlib import Path
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
task=sys.argv[1];assert task in ('cover_blocks','plug_in_charger')
from eval150_contract import output,official_config
eval_seed=int(os.environ['FLUX_EVAL_SEED']);out=output(R,task,eval_seed);out.mkdir(parents=True,exist_ok=True)
def phase(name,**kw):
    record={'time':time.time(),'phase':name,**kw}
    with (out/'phases.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
faulthandler.enable()
sys.argv=[sys.argv[0],'--allow-root']
os.environ['ROBODOJO_RUN_ID']=f'flux-r2-eval150-{task}-seed{eval_seed}'
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
                'isaacsim.replicator.behavior']
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
    cfg,effective=official_config(R,task,eval_seed,int(os.environ['FLUX_WS_PORT']))
    plan=json.loads((R/'status/r2-eval150-plan.json').read_text())['tasks'][task]['seeds'][str(eval_seed)]
    assert effective['core_sha256']==plan['effective_config_sha256']
    (out/'effective-config.json').write_text(json.dumps(effective,indent=2))
    layouts=plan['layout_ids'];assert layouts==list(range(50))
    protocol=json.loads((out/'protocol.json').read_text())
    resume_path=out/'resume.json'
    if resume_path.exists():
        assert os.environ.get('FLUX_ALLOW_RESUME')=='1','Partial evaluation requires diagnosed resume'
        resume=json.loads(resume_path.read_text())
        assert resume['task_name']==task and resume['policy_name']=='FLUX_RoboDojo_R2_Eval150'
    else:
        assert not (out/'result.json').exists(),'Missing recovery manifest; preserve evidence and diagnose'
        resume={'save_dir':str(out/'official')}
    # Preserve orphan partial-video evidence before upstream startup sweeps it.
    stream=out/'official/_stream'
    if stream.exists() and any(stream.iterdir()):
        import shutil
        evidence=out/'failure-evidence'/str(time.time_ns());evidence.mkdir(parents=True)
        shutil.move(str(stream),str(evidence/'_stream'))
    phase('environment_constructing')
    env=module.create_eval_env(cfg,app,resume_state=resume)
    for index,expected in plan['layout_sha256'].items():
        file=Path(env.seed_manager.seed_info[int(index)]['scene_layout'])
        assert hashlib.sha256(file.read_bytes()).hexdigest()==expected
    env.seed_manager.seed_list=[x for x in env.seed_manager.seed_list if x in layouts]
    env.seed_manager.ed_idx=len(env.seed_manager.seed_list)
    while env.success_nums+env.fail_nums<50:
        seeds=env.seed_manager.get_seeds(max_count=1)
        assert seeds is not None,'Official layouts exhausted before 50 counted cases'
        seed=seeds[0];assert seed in layouts
        before=env.success_nums+env.fail_nums
        (out/'IN_PROGRESS.json').write_text(json.dumps({'time':time.time(),'layout_id':seed,'completed_before':before}))
        env.reset(seed=seeds)
        # Official run_eval owns reward registration, control/deploy loop,
        # success/score calculation, video finalization and result persistence.
        env.run_eval()
        assert env.success_nums+env.fail_nums==before+1,'Uncounted/unstable case; stop, never replace layout'
        detail=env.eval_result['details'][before]
        assert detail['layout_id']==seed
        manifest=Path(env.persist_resume_manifest())
        tmp=resume_path.with_suffix('.tmp');tmp.write_bytes(manifest.read_bytes());tmp.replace(resume_path)
        (out/'result.json').write_text(json.dumps(env.eval_result,indent=2))
        (out/'progress.json').write_text(json.dumps({'time':time.time(),'completed':before+1,'successes':env.success_nums,'failures':env.fail_nums,'last_layout_id':seed},indent=2))
        (out/'IN_PROGRESS.json').unlink()
        env.seed_manager.eval_step()
        phase('official_case_complete',layout_id=seed,completed=before+1,successes=env.success_nums)
        env.close()
    assert sorted(x['layout_id'] for x in env.eval_result['details'].values())==layouts
    (out/'EVALUATION_FINISHED.json').write_text(json.dumps({'time':time.time(),'completed':50,'successes':env.success_nums,'audit_pending':True},indent=2))
    env.model_client.close();phase('finished_pending_independent_audit')
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
