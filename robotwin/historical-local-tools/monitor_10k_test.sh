set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,json,time,re,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');out={'time':time.time(),'tasks':{}}
for step in [10000]:
 for task in ['click_bell','handover_block']:
  label='flux-10k-test100';E=B/f'evaluation/{task}/{label}';p=E/'result.json';log=B/f'status/{label}-{task}.log';text=log.read_text(errors='replace') if log.exists() else '';clean=re.sub(r'\x1b\[[0-9;]*m','',text)
  scores=list(re.finditer(r'Success rate:\s*(\d+)/(\d+)\s*=>.*?current seed:\s*(\d+)',clean))
  r={'result':json.loads(p.read_text()) if p.exists() else None,'counted':list(map(int,scores[-1].groups())) if scores else None,'tail':clean[-200:]}
  if (E/'replay-implementation.json').exists() and scores:
   scored={};previous_success=previous_total=0
   for m in scores:
    success,total,seed=map(int,m.groups())
    if total==1:previous_success=previous_total=0
    if total!=previous_total+1 or success-previous_success not in [0,1] or seed in scored:raise RuntimeError('Unexpected duplicate/nonsequential official score during curated replay')
    scored[seed]=success-previous_success;previous_success,previous_total=success,total
   r['counted_last_attempt']=r['counted'];r['counted']=[sum(scored.values()),len(scored),int(scores[-1].group(3))]
  if 'Traceback' in clean:
   marker=clean.rfind('CURATED_REPLAY_BEGIN')
   latest=clean[marker:] if marker>=0 else clean
   if 'Traceback' in latest and not (r['result'] and r['result'].get('complete')):r['error']=latest[latest.index('Traceback'):][:4000]
   else:r['historical_error']=clean[clean.index('Traceback'):][:4000]
  if r['result'] and r['result'].get('complete'):
   r['counted_last_attempt']=r['counted'];r['counted']=[r['result']['successes'],r['result']['episodes'],None]
  cases=E/'cases.jsonl'
  if cases.exists():
   rows=[json.loads(x) for x in cases.read_text().splitlines() if x.strip()];r['last_case']=rows[-1] if rows else None
   accepted={int(m.group(3)) for m in scores};ends=[x for x in rows if x['event']=='trial_end' and x['seed'] in accepted];first=next((x for x in rows if x['event']=='case'),None)
   if ends and first:
    starts={x['seed']:x['time'] for x in rows if x['event']=='case'}
    durations=[x['time']-starts[x['seed']] for x in ends if x['seed'] in starts]
    mean=sum(durations)/len(durations) if durations else (ends[-1]['time']-first['time'])/len(ends)
    last=r['last_case'];elapsed=time.time()-last['time'] if last and last['event']=='case' and last['seed'] not in {x['seed'] for x in ends} else 0
    r['mean_episode_seconds']=mean;r['remaining_eta_seconds']=max(0,(100-len(ends))*mean-elapsed)
  out['tasks'][task+'-'+str(step)]=r
p=B/'status/flux-final-test-current.txt';out['current']=p.read_text().strip() if p.exists() else None
if out['current']:
 step,task,pid,port=out['current'].split();out['server_process']=subprocess.run(['ps','-p',pid,'-o','pid,etime,pcpu,pmem,args'],capture_output=True,text=True).stdout
 p=B/f'status/flux-10k-test100-{task}-server.log';s=p.read_text(errors='replace') if p.exists() else '';out['server_tail']=s[-1200:]
 if 'Traceback' in s:out['server_error']=s[s.index('Traceback'):][:5000]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
(B/'status/flux-final-test-monitor.json').write_text(json.dumps(out,indent=2))
PY
