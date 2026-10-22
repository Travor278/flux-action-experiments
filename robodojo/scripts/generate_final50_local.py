"""Derive final deployment from the verified WS interface, without training edits."""
from pathlib import Path
import ast
N=Path(__file__).parent
def write(name,text):
 if name.endswith('.py'):ast.parse(text)
 (N/name).write_text(text,encoding='utf8',newline='\n')
s=(N/'ws_probe_server.py').read_text(encoding='utf8')
s=s.replace('import asyncio, json, os, time, traceback','import asyncio, json, os, time, traceback, sys')
s=s.replace("out=Path(os.environ['FLUX_SCENE_RUN'])", "out=Path(os.environ['FLUX_SCENE_RUN']);task=sys.argv[1]\nassert task in ('cover_blocks','match_and_pick_from_conveyor')")
s=s.replace("/'runs/full-preflight-v1/export-4-ema'", "/'runs'/f'{task}-10k-prefix-v1'/'export-10000-ema'")
s=s.replace('Bounded integration server using the real export and official WS protocol.','Final10k server with the same verified CPU-first/text-offload WS deployment.')
write('serve_final50.py',s)

s=(N/'probe_ws_control.py').read_text(encoding='utf8')
s=s.replace("out=Path(os.environ['FLUX_SCENE_RUN'])/task;out.mkdir(parents=True,exist_ok=True)","out=R/'evaluation'/task/'flux-10k-prefix-v1-final50';out.mkdir(parents=True,exist_ok=True)")
s=s.replace("'control-smoke-'+out.parent.name+'-'+task", "'flux-10k-prefix-v1-final50-'+task")
s=s.replace("policy_name='FLUX_RoboDojo_History',additional_info='real_ws_smoke_not_score',seed=0", "policy_name='FLUX_RoboDojo_Final',additional_info='flux_10k_prefix_v1_final50',seed=0,eval_num=50")
s=s.replace("'policy_name':'FLUX_RoboDojo_History'", "'policy_name':'FLUX_RoboDojo_Final'")
s=s.replace('from conveyor_graph_compat import install_conveyor_graph_compat','from conveyor_graph_compat_final import install_conveyor_graph_compat')
a=s.index("    phase('environment_constructing')");b=s.index("except BaseException as exc:",a)
s=s[:a]+'''    plan=json.loads((R/'status/final50-plan.json').read_text())['tasks'][task]
    layouts=plan['layout_ids'];assert layouts==list(range(50))
    protocol=json.loads((out/'protocol.json').read_text())
    resume_path=out/'resume.json'
    if resume_path.exists():
        assert os.environ.get('FLUX_ALLOW_RESUME')=='1','Partial evaluation requires diagnosed resume'
        resume=json.loads(resume_path.read_text())
        assert resume['task_name']==task and resume['policy_name']=='FLUX_RoboDojo_Final'
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
    from conveyor_graph_compat_final import restore_loaded_conveyor_graph
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
        if task=='match_and_pick_from_conveyor':
            restore_loaded_conveyor_graph(omni.usd.get_context().get_stage(),out/'graph-compat.jsonl')
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
'''+s[b:]
s=s.replace('Real WebSocket diagnostic export integration; no benchmark score.','Fixed-layout official final50 evaluation; infrastructure failures stop without replacing scenes.')
write('run_final50.py',s)

s=(N/'conveyor_graph_compat.py').read_text(encoding='utf8')
old="        if previous and previous!=[expected]:raise RuntimeError(str(previous))"
new="""        native=graph.GetChild('read_speed_compat_5_1')
        if native and previous==[native.GetAttribute('outputs:value').GetPath()]:
            assert native.GetAttribute('node:type').Get()=='omni.graph.core.ReadVariable'
            assert native.GetAttribute('inputs:variableName').Get()=='Velocity'
            expected=native.GetAttribute('outputs:value').GetPath()
        if previous and previous!=[expected]:raise RuntimeError(str(previous))"""
assert s.count(old)==1
write('conveyor_graph_compat_final.py',s.replace(old,new))

write('deploy_final50.py','''"""Same observation whitelist and per-tick packing verified by the WS smoke."""
import numpy as np
from XPolicyLab.policy.FLUX_RoboDojo_History.model import CAMERAS,image_chw
from XPolicyLab.utils.process_data import pack_robot_state,get_robot_action_dim_info
def eval_one_episode(TASK_ENV,model_client):
    model_client.call(func_name='reset')
    info=get_robot_action_dim_info('arx_x5')
    while not TASK_ENV.is_episode_end():
        raw=TASK_ENV.get_obs()
        state=np.asarray(pack_robot_state(raw,'joint',info),dtype=np.float32)
        assert state.shape==(14,) and np.isfinite(state).all()
        images={aliases[0]:np.ascontiguousarray(image_chw(raw,aliases).transpose(1,2,0)) for aliases in CAMERAS.values()}
        model_client.call(func_name='update_obs',obs={'state':state,'images':images,'instruction':raw['instruction']})
        actions=model_client.call(func_name='get_action');assert len(actions)==1
        TASK_ENV.take_action(actions[0])
''')
