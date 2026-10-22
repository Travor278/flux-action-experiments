B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time,pickle,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));r={'time':time.time(),'scope':'failed training checkpoint and optimizer cleanup; protect live recovered training','tasks':{}}
for t in ['click_bell','handover_block']:
 R=S/f'runs/{t}-10k-tail-v1';x={}
 for p in R.glob('step-*'):
  x[p.name]={'complete':(p/'COMPLETE').exists(),'optimizer_present':(p/'optimizer').exists()}
 r['tasks'][t]=x
 r.setdefault('obsolete_failed_run_checkpoints',{})[t]=list(str(p) for p in (S/f'runs/{t}-10k-2gpu').glob('step-*'))
R=S/'runs/handover_block-10k-tail-v1';C=R/'step-100'
r['handover_old_step5_removed']=not (R/'step-5').exists()
if (C/'COMPLETE').exists():
 state=json.loads((C/'state.json').read_text());assert state['step']==100 and state['world_size']==2
 counts={}
 for part in ['model','optimizer','ema_0p10','ema_0p05']:
  with (C/part/'.metadata').open('rb') as f:m=pickle.load(f)
  assert m.state_dict_metadata;counts[part]=len(m.state_dict_metadata)
  for storage in m.storage_data.values():assert (C/part/storage.relative_path).stat().st_size >= storage.offset+storage.length
 assert len(list((C/'rng').glob('rank-*.pt')))==2
 r['handover_replacement100_verified']=counts
r['new_manual_deleted_bytes']=0
r['superseded_step5_logical_bytes']=138342594187 if r['handover_old_step5_removed'] else 0
r['free_ssd']=shutil.disk_usage(S).free
(B/'status/failed-training-cleanup-final.json').write_text(json.dumps(r,indent=2))
PY
