from pathlib import Path
import json,time,hashlib
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
plan=json.loads((R/'status/selected-download-plan.json').read_text());meta=json.loads((R/'status/official-v3-inspection.json').read_text());prefix=meta['prefix']
specs=[f for f in plan['files'] if f['path'].startswith(prefix+'/data/')]
ids=sum(plan['selected_episode_ids'].values(),[]);tables=[];sources={}
for spec in specs:
 p=R/'official-source'/spec['path']
 if not p.exists() or p.stat().st_size!=spec['bytes']:
  (R/'status/numeric-audit-waiting.json').write_text(json.dumps({'time':time.time(),'missing':str(p)}));raise SystemExit(0)
 with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 assert digest==spec['lfs_sha256'];sources[spec['path']]=digest
 tables.append(pq.read_table(p,columns=['episode_index','frame_index','timestamp','observation.state','action','task_index'],filters=[('episode_index','in',ids)]))
table=pa.concat_tables(tables);out={'time':time.time(),'revision':meta['revision'],'source_sha':sources,'tasks':{},'scope':'Numeric action/state/time only; RGB decode and semantic alignment still require independent validation'}
for task,selected in meta['selected'].items():
 states=[];actions=[];episodes=[]
 for ep in selected['rows']:
  e=table.filter(pa.compute.equal(table['episode_index'],ep['episode_index']))
  order=np.argsort(e['frame_index'].to_numpy());e=e.take(pa.array(order))
  assert e.num_rows==ep['length'];f=e['frame_index'].to_numpy();assert np.array_equal(f,np.arange(len(f)))
  t=e['timestamp'].to_numpy();assert np.allclose(t,np.arange(len(t))/25,atol=1e-4,rtol=0)
  s=np.asarray(e['observation.state'].to_pylist(),np.float32);a=np.asarray(e['action'].to_pylist(),np.float32)
  assert s.shape==a.shape==(ep['length'],14) and np.isfinite(s).all() and np.isfinite(a).all()
  assert len(set(e['task_index'].to_pylist()))==1
  states.append(s);actions.append(a);episodes.append({'episode_index':ep['episode_index'],'frames':len(f)})
 s=np.concatenate(states);a=np.concatenate(actions)
 out['tasks'][task]={'episodes':episodes,'count':len(episodes),'frames':len(a),'state_q01':np.quantile(s,.01,axis=0).tolist(),'state_q99':np.quantile(s,.99,axis=0).tolist(),'action_q01':np.quantile(a,.01,axis=0).tolist(),'action_q99':np.quantile(a,.99,axis=0).tolist(),'action_min':a.min(0).tolist(),'action_max':a.max(0).tolist()}
 assert len(episodes)==100
out['complete']=True;(R/'status/numeric-data-verified.json').write_text(json.dumps(out,indent=2));print('NUMERIC_200_EPISODES_PASS',flush=True)
