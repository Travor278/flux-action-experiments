import pathlib,json,time,shutil,os
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
proof=json.loads((B/'status/optimized-final-artifacts-verified.json').read_text());targets=[]
for task in ['click_bell','handover_block']:
 assert proof['tasks'][task]['verified'] and len(proof['tasks'][task]['exports'])==4
 R=S/f'runs/{task}-10k-opt-v1';assert (R/'TRAIN_COMPLETE.json').exists()
 targets.append((task,R/'step-10000'))
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:cmd=(p/'cmdline').read_bytes().decode(errors='replace')
 except OSError:continue
 assert not ('scripts/train_efficiency_worker.py' in cmd and str(B) in cmd),'Own worker still active'
record={'time':time.time(),'request':'User requested deleting process checkpoints and optimizer after final export verification','deleted':[]}
for task,c in targets:
 if not c.exists():continue
 assert c.resolve()==S.resolve()/f'runs/{task}-10k-opt-v1/step-10000' and not c.is_symlink()
 state=json.loads((c/'state.json').read_text());assert state['step']==10000 and (c/'COMPLETE').exists()
 evidence=B/'status/optimized-final-checkpoint-metadata'/task;evidence.mkdir(parents=True,exist_ok=True)
 for name in ['state.json','config.json','COMPLETE']:shutil.copy2(c/name,evidence/name)
 files=[f for f in c.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in files)
 shutil.rmtree(c);record['deleted'].append({'path':str(c),'files':len(files),'bytes':size})
(B/'status/optimized-training-state-cleanup.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
