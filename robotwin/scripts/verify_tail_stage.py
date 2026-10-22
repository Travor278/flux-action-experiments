import pathlib,json,hashlib,sys,time,pickle,math
from safetensors import safe_open
from tail_log_records import log_dicts
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
task,step=sys.argv[1],int(sys.argv[2]);assert task in ['click_bell','handover_block'] and step in [4,5,100,10000]
R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/f'runs/{task}-10k-tail-v1'
C=R/f'step-{step}';assert (C/'COMPLETE').is_file()
state=json.loads((C/'state.json').read_text());assert state['step']==step and state['world_size']==2
key_counts={}
for part in ['model','optimizer','ema_0p10','ema_0p05']:
 with (C/part/'.metadata').open('rb') as f: metadata=pickle.load(f)
 assert metadata.state_dict_metadata,part
 key_counts[part]=len(metadata.state_dict_metadata)
 for storage in metadata.storage_data.values():
  file=C/part/storage.relative_path
  assert file.is_file() and file.stat().st_size>=storage.offset+storage.length
assert len(list((C/'rng').glob('rank-*.pt')))==2,'Two rank RNG states required'
logs=R/'train.log'
records=[r for r in log_dicts(logs.read_text(errors='replace')) if r.get('event')=='EFFICIENCY_RANK_READY']
if step==5:
 assert {r['rank'] for r in records if r.get('step')==4}=={0,1},'Both ranks must restore4 before step5'
metrics=[json.loads(x) for x in (R/'metrics.jsonl').read_text().splitlines()]
for m in metrics:
 for k in ['loss','grad_norm']:
  if k in m: assert math.isfinite(m[k]),(k,m)
out={'time':time.time(),'task':task,'step':step,'complete':True,'dcp_keys':key_counts,'rank_ready':records}
(R/f'stage-{step}-verified.json').write_text(json.dumps(out,indent=2))
if '--export' in sys.argv:
 E=R/'export-10000-ema';M=json.loads((E/'manifest.json').read_text())
 for n,h in M['sha256'].items():
  with (E/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
 with safe_open(str(E/'model.safetensors'),framework='pt',device='cpu') as f:assert len(f.keys())==451
 out['export']=str(E);out['manifest']=M;(R/'TRAIN_COMPLETE.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'event':'TAIL_STAGE_VERIFIED','task':task,'step':step,'export':'--export' in sys.argv}),flush=True)
