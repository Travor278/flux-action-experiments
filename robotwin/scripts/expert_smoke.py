import json,pathlib,sys,time
import numpy as np
B=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'RoboTwin/scripts'))
from eval_policy_xpolicylab import load_task_args,class_decorator,robotwin_obs_to_xpolicylab

def main():
    results={}
    for task in ['click_bell','handover_block']:
        args,embodiment=load_task_args({'task_name':task,'task_config':'demo_clean','policy_name':'expert_smoke'})
        args.update(collect_data=False,save_data=False,render_freq=0,eval_video_log=False,save_path=str(B/'status/expert-smoke'))
        env=class_decorator(task);started=time.perf_counter()
        try:
            env.setup_demo(now_ep_num=0,seed=100000,is_test=True,**args)
            obs=robotwin_obs_to_xpolicylab(env.get_obs(),instruction='expert environment smoke',env_idx=0,frequency=15,task_env=env)
            images={k:list(v['color'].shape) for k,v in obs['vision'].items() if 'color' in v}
            assert all(np.isfinite(v['color']).all() for v in obs['vision'].values() if 'color' in v)
            env.play_once()
            result={'embodiment':embodiment,'seed':100000,'planner':args['left_embodiment_config']['planner'],
                    'plan_success':bool(env.plan_success),'task_success':bool(env.check_success()),
                    'image_shapes':images,'seconds':time.perf_counter()-started,
                    'scope':'scripted expert smoke only; not a learned-policy evaluation'}
            results[task]=result;print(json.dumps({task:result}),flush=True)
            (B/'status/expert-smoke.json').write_text(json.dumps(results,indent=2))
        finally:env.close_env(clear_cache=True)

if __name__=='__main__':main()
