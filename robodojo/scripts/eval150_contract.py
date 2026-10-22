"""Frozen evaluation-only contract; no training imports or parameter changes."""
import ast,hashlib,json,math,os,statistics
from pathlib import Path
from types import SimpleNamespace
TASKS=('cover_blocks','plug_in_charger')
SEEDS=(0,1,2)
POLICY='FLUX_RoboDojo_R2_Eval150'
TAG='flux-10k-official-history-r2-20261021-eval150'
def output(root,task,seed):
    assert task in TASKS and seed in SEEDS
    return Path(root)/'evaluation'/task/TAG/f'seed-{seed}'
def validate_result(result,layout_ids):
    assert result['eval_time']==len(layout_ids)==50
    details=sorted(result['details'].items(),key=lambda x:int(x[0]))
    assert [int(k) for k,v in details]==list(range(50))
    assert [v['layout_id'] for k,v in details]==layout_ids
    assert len(set(layout_ids))==50
    scores=[];success=0
    for _,v in details:
        assert isinstance(v['success'],bool)
        score=float(v['score']);assert math.isfinite(score) and 0<=score<=1
        if v['success']:assert abs(score-1)<1e-12
        success+=int(v['success']);scores.append(score)
    mean_score=statistics.mean(scores)*100
    assert math.isfinite(result['success_rate']) and abs(result['success_rate']-success/50)<1e-12
    assert math.isfinite(result['score']) and abs(result['score']-mean_score)<1e-8
    return {'successes':success,'total':50,'success_rate':success/50,'score':mean_score}
def aggregate(rows):
    assert sorted(x['eval_seed'] for x in rows)==list(SEEDS)
    assert len({x['model_sha256'] for x in rows})==1,'Same checkpoint across all seeds'
    assert len({x['plan_sha256'] for x in rows})==1
    assert all(x['verified'] and x['total']==50 for x in rows)
    sr=[x['success_rate']*100 for x in rows];scores=[x['score'] for x in rows]
    return {'runs':3,'total':150,'successes':sum(x['successes'] for x in rows),
            'success_rate_percent_mean':statistics.mean(sr),'success_rate_percent_std':statistics.pstdev(sr),
            'score_mean':statistics.mean(scores),'score_std':statistics.pstdev(scores),
            'std_ddof':0,'sample_std_ddof1':{'success_rate_percent':statistics.stdev(sr),'score':statistics.stdev(scores)},
            'scope':'One frozen training checkpoint, three official evaluation layout seeds; NOT three independent training runs'}
def official_config(root,task,seed,port):
    """Execute only the pinned official main() configuration block, without AppLauncher."""
    import importlib
    from omegaconf import OmegaConf
    from env.global_configs import ROOT_DIR,ENV_CONFIG_PATH,BENCHMARK
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_randomization,process_config,resolve_random_task_num_envs
    assert task in TASKS and seed in SEEDS
    assert os.environ.get('EVAL_NUM') in (None,'50','native')
    BENCHMARK_PATH=os.path.join(ROOT_DIR,'task',BENCHMARK)
    path=Path(root)/'RoboDojo/src/eval_client/main.py';tree=ast.parse(path.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    body=[]
    for node in main.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='run_id' for t in node.targets):break
        body.append(node)
    assert body and 'env_cfg.sim.seed' in ast.unparse(body[-1]),'Official config block changed'
    fn=ast.FunctionDef(name='_assemble',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body+[ast.Return(value=ast.Tuple(elts=[ast.Name(id='env_cfg',ctx=ast.Load()),ast.Name(id='eval_num',ctx=ast.Load())],ctx=ast.Load()))],decorator_list=[])
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_eval_batch_from_deploy')
    task_cfg=load_yaml(str(Path(ROOT_DIR)/'task'/BENCHMARK/'config'/f'{task}.yml'))
    assert not task_cfg.get('Articulation'),'Add official PhysX recovery integration before using articulated tasks'
    args=SimpleNamespace(task_name=task,num_envs=1,env_cfg_type='arx_x5',device_id=0,policy_name=POLICY,port=port,host='127.0.0.1',protocol='ws',policy_server_url='',additional_info=TAG,seed=seed)
    g=locals().copy();g.update(os=os,task_registry=importlib.import_module(f'task.{BENCHMARK}.task_registry'),args_cli=args,enable_monitor=False)
    module=ast.fix_missing_locations(ast.Module(body=[helper,fn],type_ignores=[]));exec(compile(module,str(path),'exec'),g)
    cfg,number=g['_assemble']();assert number==50 and cfg.eval_cfg.seed==seed and cfg.sim.scene.num_envs==1
    assert cfg.camera.default_frequency==25 and cfg.sim.get('device','cpu')=='cpu'
    assert list(cfg.sim.seed)==[0] # Official initializer; env.reset later uses layout IDs.
    value=OmegaConf.to_container(cfg,resolve=True);core={k:value[k] for k in ['sim','scene','camera','robot','task_env','eval_cfg']}
    digest=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return cfg,{'config':value,'core_sha256':digest,'official_main_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'assembled_from_official_ast':True}
