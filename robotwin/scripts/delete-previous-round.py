from pathlib import Path
import json,shutil,time,re
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008').resolve()
S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008').resolve()
old={'click_bell':'job-REPLACE_WITH_YOUR_JOB_ID','handover_block':'job-REPLACE_WITH_YOUR_JOB_ID'}
targets=[S/'runs'/t for t in old]+[B/'runs/invalid-profile-shadowing-20261008',B/'runs/checkpoint-cpu-smoke']
assert 'CPU_2RANK_MODEL_OPTIMIZER_DUAL_EMA_SAVE_RESTORE_PASS' in (B/'status/checkpoint-cpu-test.log').read_text()
assert (B/'status/checkpoint-causal-warnings.json').is_file()
receipt={'time':time.time(),'removed':[],'previous_last_steps':{},'protected_repair_path':str(S/'runs/checkpoint-repair-v1')}
for task,job in old.items():
 p=S/'runs'/task
 if not p.exists():continue
 assert p.resolve()==p and not p.is_symlink()
 assert not list(p.rglob('COMPLETE')),'Unexpected complete training checkpoint'
 text=(p/'train.log').read_text(errors='replace');ids=set(re.findall(r'job-[a-f0-9-]{36}',text));assert ids=={job},ids
 rows=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines() if x.strip()];receipt['previous_last_steps'][task]=[x for x in rows if 'loss' in x][-1]
for p in targets:
 if not p.exists():continue
 assert p.resolve()==p and not p.is_symlink() and (p.is_relative_to(B/'runs') or p.is_relative_to(S/'runs'))
 assert all(not x.is_symlink() for x in p.rglob('*'))
 files=[x for x in p.rglob('*') if x.is_file()];receipt['removed'].append({'path':str(p),'files':len(files),'bytes':sum(x.stat().st_size for x in files),'allocated_bytes':sum(x.stat().st_blocks*512 for x in files)})
 shutil.rmtree(p)
for name in ['eta.json','formal-monitor.json','formal-monitor.log','cleanup-audit.json']:
 p=B/'status'/name
 if p.is_file() and not p.is_symlink():
  st=p.stat();receipt['removed'].append({'path':str(p),'files':1,'bytes':st.st_size,'allocated_bytes':st.st_blocks*512});p.unlink()
receipt['all_targets_absent']=all(not p.exists() for p in targets)
receipt['total_files']=sum(x['files'] for x in receipt['removed']);receipt['total_bytes']=sum(x['bytes'] for x in receipt['removed']);receipt['total_allocated_bytes']=sum(x['allocated_bytes'] for x in receipt['removed'])
(B/'status/previous-round-cleanup-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
