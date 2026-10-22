from pathlib import Path
import json,hashlib,sys,time,copy
import numpy as np
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');S=Path(str(R).replace('/hdd/','/ssd/'))
sys.path.insert(0,str(R/'flux-action-official-v1/src'))
from flux_action.training.data import WindowDataset
from flux_action.training.trainer import TrainConfig,Trainer
from flux_action.config import PolicyConfig
task=sys.argv[1];assert task in ['cover_blocks','plug_in_charger']
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
old=json.loads((R/'status/official-history-ready.json').read_text())
for n,h in old['files'].items():assert sha(R/n)==h,n
cfg=json.loads((R/'configs/cover_blocks-10k-official-history-v1.json').read_text())
cfg['output_dir']=str(S/'runs'/f'{task}-10k-official-history-r2-20261021')
assert not Path(cfg['output_dir']).exists(),'Never overwrite or reprepare an existing run'
if task=='plug_in_charger':
 import pyarrow.parquet as pq
 import pyarrow as pa
 import av
 verified=json.loads((R/'status/charger-download-verified-20261021.json').read_text());assert verified['complete']
 cfg['source_root']=verified['source'];root=Path(verified['source'])
 eps=json.loads((R/'status/charger-download-plan-20261021.json').read_text())['episodes'];assert [e['episode_index'] for e in eps]==list(range(2200,2300))
 idx=R/'assets/official-history-r2-20261021'/task;idx.mkdir(parents=True,exist_ok=True)
 mapping={'head':'cam_high','left_wrist':'cam_left_wrist','right_wrist':'cam_right_wrist'}
 m=copy.deepcopy(json.loads((R/'assets/official-history-index-v1/cover_blocks/manifest.json').read_text()));m.update(robodojo_task=task,episodes=[])
 tables={};arrays=[];offset=0;decode_count=0
 for rel,rec in verified['files'].items():assert sha(root/rel)==rec['sha256']
 for ep in eps:
  rel=f"{m['source_prefix']}/data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet"
  if rel not in tables:tables[rel]=pq.read_table(root/rel,columns=['episode_index','frame_index','timestamp','observation.state','action'])
  t=tables[rel];part=t.filter(pa.compute.equal(t['episode_index'],ep['episode_index'])).sort_by('frame_index');length=ep['length']
  assert part['frame_index'].to_pylist()==list(range(length)) and np.allclose(part['timestamp'].to_numpy(),np.arange(length)/25,atol=1e-4,rtol=0)
  state=np.asarray(part['observation.state'].to_pylist(),np.float32);action=np.asarray(part['action'].to_pylist(),np.float32)
  assert state.shape==action.shape==(length,14) and np.isfinite(state).all() and np.isfinite(action).all()
  arrays.append(np.concatenate([state,action],axis=1));videos={};available=length
  for name,cam in mapping.items():
   k='videos/observation.images.'+cam;first=round(ep[k+'/from_timestamp']*25);count=round((ep[k+'/to_timestamp']-ep[k+'/from_timestamp'])*25)
   assert 0<=length-count<=1;available=min(available,count)
   rel=f"{m['source_prefix']}/{k}/chunk-{ep[k+'/chunk_index']:03d}/file-{ep[k+'/file_index']:03d}.mp4"
   with av.open(str(root/rel)) as c:
    st=c.streams.video[0];st.thread_count=1;assert float(st.average_rate)==25;c.seek(int((first/25)/st.time_base),stream=st,backward=True,any_frame=False);seen=0
    for f in c.decode(st):
     pts=float(f.pts*st.time_base)
     if pts<first/25-.001:continue
     assert abs(pts-(first+seen)/25)<.001 and (f.width,f.height)==(640,480)
     if seen in (0,count//2,count-1):assert f.to_ndarray(format='rgb24').std()>1
     seen+=1
     if seen==count:break
    assert seen==count
   decode_count+=1;videos[name]={'file':rel,'first_frame':first}
  m['episodes'].append({'episode_index':ep['episode_index'],'episode_id':str(ep['episode_index']),'from_index':offset,'n_frames':length,'caption':' | '.join(ep['tasks']),'split':'train','valid_ranges':[[1,available]],'videos':videos,'source_episode':ep});offset+=length
  if len(m['episodes'])%10==0:print('CHARGER_DECODED_EPISODES',len(m['episodes']),flush=True)
 rows=np.concatenate(arrays);np.save(idx/'rows.f32.npy',rows)
 m.update(total_frames=offset,rows={'file':'rows.f32.npy','width':28,'sha256':sha(idx/'rows.f32.npy')});(idx/'manifest.json').write_text(json.dumps(m,indent=2))
 for key,data in [('state',rows[:,:14]),('action',rows[:,14:])]:cfg['policy'][key+'_normalization']={'q01':np.quantile(data,.01,axis=0).tolist(),'q99':np.quantile(data,.99,axis=0).tolist()}
 stats={'format_version':1,'normalization':'range','clip':6.0,'episodes':100,'frames':offset,'action':{'parameterization':'absolute','absolute_dims':[],**cfg['policy']['action_normalization']},'state':cfg['policy']['state_normalization']}
 (idx/'statistics.json').write_text(json.dumps(stats,indent=2));cfg['index_dir']=str(idx)
 assert decode_count==300
idx=Path(cfg['index_dir']);m=json.loads((idx/'manifest.json').read_text())
pc=PolicyConfig(**cfg['policy']);pc.validate_training();assert not hasattr(pc,'prefix_memory_spec')
tc=TrainConfig(**cfg);assert tc.global_batch(2)==32 and tc.steps==10000
dummy=object.__new__(Trainer);dummy.manifest=m;dummy._log_line=lambda r:None;Trainer._reconcile_with_index(dummy,cfg['policy'],idx)
ds=WindowDataset(m,cfg['source_root'],idx/m['rows']['file'],seed=42,n_obs_steps=8,decoder='pyav',frame_hw=(480,640),visits_per_epoch=64,num_workers=2,world_size=2,windows_per_rank=2,grad_accumulation=8)
assert len(ds.episodes)==100
decoded=[]
for i in [0,24,99]:
 for start in [ds.valid_starts(i)[0],ds.valid_starts(i)[-1]]:
  item=ds.window_at(i,start);ep=ds.episodes[i];rows=ds.rows[ep['from_index']:ep['from_index']+ep['n_frames']]
  assert np.array_equal(item['state'],rows[start-7:start+1,:14]) and np.array_equal(item['command_history'],rows[start-8:start,14:]) and np.array_equal(item['action'],rows[start:start+32,14:])
  assert not any('prefix' in k or 'mask' in k for k in item);decoded.append([ep['episode_index'],start])
path=R/f'configs/{task}-10k-official-history-r2-20261021.json';path.write_text(json.dumps(cfg,indent=2))
files={n:h for n,h in old['files'].items() if n.startswith('flux-action-official-v1/src/')}
for n in ['train_official_history.py','async_cpu_ema.py','cpu_power_ema.py','launch_train_official_r2.sh','audit_official_r2.py','check_official_r2_gate.py']:files['scripts/'+n]=sha(R/'scripts'/n)
for p in [path,idx/'manifest.json',idx/'statistics.json']:files[str(p.relative_to(R))]=sha(p)
result={'time':time.time(),'ready':True,'fresh_base_optimizer':True,'official_commit':old['official_commit'],'custom_prefix':False,'task':task,'episodes':100,'decoded_windows':decoded,'full_decode_spans':300 if task=='plug_in_charger' else 'prior600auditpassed','eligible_starts':sum(len(ds.valid_starts(i)) for i in range(100)),'config':str(path),'output':cfg['output_dir'],'files':files}
(R/f'status/official-r2-{task}-ready.json').write_text(json.dumps(result,indent=2));print('OFFICIAL_R2_READY',task,flush=True)
