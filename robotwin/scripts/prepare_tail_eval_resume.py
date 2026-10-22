import pathlib,json,re,hashlib,time,shutil,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');E=B/'evaluation/handover_block/flux-tail-10k-test100';log=B/'status/flux-tail-10k-test100-handover_block.log'
assert not (E/'result.json').exists()
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try:c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  assert not (str(B) in c and ('setup_policy_server.py' in c or 'run_eval_paired.py' in c or 'launch_flux_tail_test100.sh' in c)),c
text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text(errors='replace'));scores=[tuple(map(int,x)) for x in re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>.*?current seed:\s*(\d+)',text)]
assert len(scores)==33 and scores[-1]==(9,33,100039)
rows=[json.loads(x) for x in (E/'cases.jsonl').read_text().splitlines() if x.strip()];cases={x['seed']:x for x in rows if x['event']=='case'};ends={x['seed']:x for x in rows if x['event']=='trial_end'}
prev=0;preserved=[]
for i,(succ,total,seed) in enumerate(scores,1):
 assert i==total and seed in ends and bool(succ-prev)==bool(ends[seed]['success']);preserved.append({**cases[seed],'episode':i,'success':bool(succ-prev)});prev=succ
assert len({x['seed'] for x in preserved})==33
assert rows[-1]['event']=='case' and rows[-1]['seed']==100040
backup=B/'status/tail-handover-resume-evidence';backup.mkdir(exist_ok=False)
sha={}
for p in [E/'cases.jsonl',E/'protocol.json',log,B/'status/flux-tail-10k-test100-handover_block-server.log',B/'status/flux-tail-final-test-launch.log']:
 sha[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,backup/p.name)
V=B/'RoboTwin/eval_result/handover_block/FLUX_Action/demo_clean/flux-tail-10k-test100';videos=list(V.rglob('*.mp4'));retained=[];partial=[]
for p in videos:
 m=re.fullmatch(r'episode(\d+)\.mp4',p.name);assert m,p
 i=int(m[1]);assert i<=33
 if i<33:
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(p)],text=True));assert float(info['format']['duration'])>0;retained.append(i)
 else:
  target=backup/('unscored-'+p.name);assert not target.exists();shutil.move(str(p),str(target));partial.append({'from':str(p),'to':str(target)})
assert sorted(retained)==list(range(33))
plan={'time':time.time(),'retained_episodes':33,'retained_successes':9,'next_seed':100040,'total_episodes':100,'incomplete_instruction':cases[100040]['instruction'],'retained_pairs':preserved,'prefix_sha256':sha,'partial_video_archived':partial,'expert_check':True,'reason':'Process disappeared mid-rollout without traceback; exact cause unknown. Retain all officially scored outcomes, retry only unfinished seed.'}
(B/'status/tail-handover-resume-plan.json').write_text(json.dumps(plan,indent=2));print('RESUME_PREFLIGHT_PASS',len(preserved))
