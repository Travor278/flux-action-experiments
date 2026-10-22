"""Diagnostic replay of recorded expert commands, never a policy benchmark score."""
import argparse, gc, hashlib, importlib.util, json, os, pathlib, subprocess, sys, time, traceback
import numpy as np
import torch

B = pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
R = B / 'evaluation/expert-replay-click'
os.chdir(B / 'RoboTwin')
for p in [B, B/'RoboTwin', B/'XPolicyLab']:
    sys.path.insert(0, str(p))
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.12, 0)
spec = importlib.util.spec_from_file_location('expert_replay_upstream', B/'RoboTwin/scripts/eval_policy_xpolicylab.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=1)
opt = parser.parse_args()
plan = json.loads((R/'plan.json').read_text())
args, _ = upstream.load_task_args({'task_name':'click_bell', 'task_config':'demo_clean',
    'policy_name':'ExpertReplayDiagnostic', 'ckpt_setting':'recorded-source-native14'})
args.update(save_data=False, collect_data=False, render_freq=0, eval_mode=True,
            need_plan=True, save_path=str(R/'sim-cache'))
keys = ['left_arm_joint_state', 'left_ee_joint_state', 'right_arm_joint_state', 'right_ee_joint_state']
edges = [0,6,7,13,14]

def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)); temp.replace(path)

def native_state(obs):
    x = np.asarray(obs['joint_action']['vector'], dtype=np.float32)
    assert x.shape == (14,) and np.isfinite(x).all()
    return x

def setup(record):
    env = upstream.class_decorator('click_bell')
    env.setup_demo(now_ep_num=record['episode'], seed=record['seed'], is_test=True, **args)
    env.step_lim = 400
    return env

def close(env):
    try: env.close_env(clear_cache=True)
    except Exception: pass
    for name in ['robot','cameras','scene']:
        if hasattr(env,name): setattr(env,name,None)
    gc.collect()
    torch.cuda.empty_cache()

def replay(record):
    outdir = R/f'episode-{record["episode"]}'
    outdir.mkdir(exist_ok=True)
    saved = outdir/'result.json'
    if saved.exists(): return json.loads(saved.read_text())
    raw = np.load(record['npz'])
    env = None
    started = time.time()
    result = {'episode':record['episode'], 'seed':record['seed'], 'task':'click_bell',
              'source_hdf5_sha256':record['source_hdf5_sha256'], 'started':started,
              'recorded_frames':len(raw['action']), 'model_used':False, 'diagnostic_only':True}
    video = None
    try:
        env = setup(record)
        obs = env.get_obs()
        initial_error = float(np.max(np.abs(native_state(obs)-raw['state'][0])))
        expected_arm = record['source_scene']['info']['{a}']
        expected_id = int(record['source_scene']['info']['{A}'].rsplit('base',1)[1])
        actual_arm = 'left' if env.bell.get_pose().p[0] < 0 else 'right'
        camera_errors = {}
        for source, runtime in [('cam_head','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]:
            cam = obs['observation'][runtime]
            a = np.asarray(cam['rgb'], dtype=np.uint8)
            runtime_e=np.asarray(cam['extrinsic_cv'])[:3,:4]
            source_e=raw[source+'_extrinsic0'][:3,:4]
            axis_flip=np.diag([1.,-1.,-1.,1.])
            camera_errors[source] = {
                'intrinsic_max_error':float(np.max(np.abs(cam['intrinsic_cv']-raw[source+'_intrinsic0']))),
                'extrinsic_max_error':float(np.max(np.abs(runtime_e-source_e))),
                'extrinsic_after_world_yz_axis_flip_max_error':float(np.max(np.abs(runtime_e-source_e@axis_flip))),
                'runtime_extrinsic':runtime_e.tolist(),'source_extrinsic':source_e.tolist(),
                'rgb_mean_abs_error':float(np.mean(np.abs(a.astype(float)-raw[source+'_rgb0'].astype(float)))),
                'shape':list(a.shape)}
        result.update(initial_state_max_error=initial_error, camera_errors=camera_errors,
                      source_arm=expected_arm, runtime_arm=actual_arm, source_bell_id=expected_id,
                      runtime_bell_id=int(env.bell_id), bell_pose=env.bell.get_pose().p.tolist())
        result['scene_identity_matches'] = expected_arm==actual_arm and expected_id==int(env.bell_id)
        if not result['scene_identity_matches']:
            result['status']='scene_identity_mismatch';result['replay_success']=None
        else:
            h,w=obs['observation']['head_camera']['rgb'].shape[:2]
            video=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
                '-video_size',f'{w}x{h}','-framerate','10','-i','-','-threads','1','-pix_fmt','yuv420p',
                '-vcodec','libx264','-crf','23',str(outdir/'replay.mp4')],stdin=subprocess.PIPE)
            errors=[];success=False
            for i, command in enumerate(raw['action']):
                before=native_state(obs)
                action={k:command[edges[j]:edges[j+1]].copy() for j,k in enumerate(keys)}
                flat,kind=upstream.xpolicylab_action_to_robotwin(action,action_type='joint',current_observation=obs)
                assert kind=='qpos' and np.array_equal(flat.astype(np.float32),command)
                video.stdin.write(np.asarray(obs['observation']['head_camera']['rgb'],dtype=np.uint8).tobytes())
                env.take_action(flat,action_type=kind)
                obs=env.get_obs();after=native_state(obs)
                e={'frame':i,'time':time.time(),'pre_state_vs_source_max':float(np.max(np.abs(before-raw['state'][i]))),
                   'post_state_vs_command_max':float(np.max(np.abs(after-command))),
                   'left_gripper':float(after[6]),'right_gripper':float(after[13]),
                   'success':bool(env.eval_success or env.check_success())}
                errors.append(e)
                with (outdir/'trace.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
                write_json(B/'status/expert-replay-progress.json',{'time':time.time(),'episode':record['episode'],
                    'seed':record['seed'],'frame':i,'total_frames':len(raw['action']),'success':e['success'],'phase':'recorded_replay'})
                if e['success']:success=True;break
            video.stdin.write(np.asarray(obs['observation']['head_camera']['rgb'],dtype=np.uint8).tobytes())
            video.stdin.close();assert video.wait(timeout=30)==0;video=None
            result.update(status='recorded_replay_complete',replay_success=success,
                          executed_frames=len(errors),first_success_frame=errors[-1]['frame'] if success else None,
                          max_command_tracking_error=max(e['post_state_vs_command_max'] for e in errors),
                          mean_command_tracking_error=float(np.mean([e['post_state_vs_command_max'] for e in errors])))
        close(env);env=None
        # Control group: regenerate the current simulator's own expert on the same seed.
        if result.get('replay_success') is not True:
            env=setup(record);env.play_once()
            result['fresh_expert_control']={'plan_success':bool(env.plan_success),'success':bool(env.check_success())}
            close(env);env=None
        result['complete']=True;result['finished']=time.time();result['wall_seconds']=result['finished']-started
        write_json(saved,result)
        return result
    except Exception:
        result.update(status='exception',error=traceback.format_exc(),finished=time.time(),complete=False)
        write_json(outdir/'error.json',result)
        raise
    finally:
        if video is not None:
            try:video.stdin.close();video.wait(timeout=10)
            except Exception:video.kill()
        if env is not None:close(env)

results=[]
try:
    for record in plan['episodes'][:opt.limit]:
        results.append(replay(record))
        write_json(B/'status/expert-replay-summary.json',{'time':time.time(),'requested':opt.limit,
            'completed':len(results),'recorded_successes':sum(r.get('replay_success') is True for r in results),
            'results':results,'complete':len(results)==opt.limit})
    print(json.dumps({'complete':True,'episodes':len(results),'recorded_successes':sum(r.get('replay_success') is True for r in results)}),flush=True)
except Exception:
    write_json(B/'status/expert-replay-failure.json',{'time':time.time(),'error':traceback.format_exc()})
    raise
