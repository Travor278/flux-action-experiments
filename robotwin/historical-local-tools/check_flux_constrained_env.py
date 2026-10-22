import base64,hashlib,importlib.metadata as m,json,pathlib,sys,time,traceback
import torch
W=pathlib.Path('/home/USER/rpent-data/codex_remote_ops/flux_env_compat_20261010')
out={'time':time.time(),'python':sys.version,'env':sys.prefix,'versions':{n:m.version(n) for n in ['torch','torchvision','lerobot','transformers','pyarrow','av','torchcodec']},'checks':{}}
def check(name,fn):
    try: out['checks'][name]={'pass':True,'detail':fn()}
    except Exception: out['checks'][name]={'pass':False,'error':traceback.format_exc()}
def wheel_integrity():
    result={}
    for name in ['lerobot','transformers']:
        d=m.distribution(name); files=[f for f in d.files if str(f).endswith('.py') and f.hash]
        bad=[]
        for f in files:
            actual=base64.urlsafe_b64encode(hashlib.sha256(d.locate_file(f).read_bytes()).digest()).rstrip(b'=').decode()
            if actual!=f.hash.value: bad.append(str(f))
        assert not bad,bad
        result[name]={'verified_python_files':len(files),'modified':bad}
    return result
def cuda():
    x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    (x@x).float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    return {'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'bf16_forward_backward':True}
def qwen_api():
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration,AutoProcessor
    cfg=Qwen3VLConfig(text_config={'vocab_size':128,'hidden_size':128,'intermediate_size':256,'num_hidden_layers':2,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':32,'rope_parameters':{'rope_type':'default','rope_theta':500000,'mrope_section':[4,6,6],'mrope_interleaved':True}},vision_config={'depth':1,'hidden_size':32,'intermediate_size':64,'num_heads':4,'out_hidden_size':128,'deepstack_visual_indexes':[]},image_token_id=124,video_token_id=125,vision_start_token_id=126,vision_end_token_id=127)
    model=Qwen3VLForConditionalGeneration(cfg).eval()
    with torch.no_grad(): r=model.model(input_ids=torch.randint(0,100,(2,8)),attention_mask=torch.ones(2,8,dtype=torch.long),output_hidden_states=True,use_cache=False)
    assert len(r.hidden_states)==3
    assert torch.isfinite(torch.stack(r.hidden_states)).all()
    return {'hidden_state_shapes':[list(x.shape) for x in r.hidden_states],'note':'Random small Qwen, exact text-only model.model call used by FLUX; not pretrained encoder parity'}
def actual_v3():
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from flux_action.data.lerobot.index import index_dataset,load_manifest
    from flux_action.training.data import WindowDataset
    results={}
    for task in ['click_bell','handover_block']:
        root=pathlib.Path('/home/USER/codex_remote_ops/flux_robotwin_pi05_20261008/assets/pi05-v3')/task
        info=json.loads((root/'meta/info.json').read_text()); assert info['codebase_version']=='v3.0'
        metadata=LeRobotDatasetMetadata(repo_id='audit/'+task,root=root)
        cameras={k.removeprefix('observation.images.'):k for k,v in info['features'].items() if v['dtype']=='video'}
        summary=index_dataset(root,W/('index-'+task),cameras,action_parameterization='absolute',absolute_action_dims=(),chunk_size=32)
        man=load_manifest(W/('index-'+task)/'manifest.json')
        ds=WindowDataset(man,root,W/('index-'+task)/'rows.f32.npy',seed=42,decoder='pyav',frame_hw=(240,320))
        sample=ds.window(0)
        assert sample['action'].shape==(32,14)
        return_sample={k:list(v.shape) for k,v in sample.items() if isinstance(v,torch.Tensor)}
        results[task]={'official_metadata_episodes':metadata.total_episodes,'source_version':info['codebase_version'],'flux_sample':return_sample}
    return results
check('official_wheel_integrity',wheel_integrity)
check('cuda_bf16',cuda)
check('transformers_qwen_interface',qwen_api)
check('real_lerobot_v3_flux_read',actual_v3)
out['scope']='Compatibility probes, not full pretrained 7B training or checkpoint parity'
(W/'status/compatibility.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
