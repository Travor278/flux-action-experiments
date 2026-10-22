import os,signal,time,pathlib,json,re,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
def proc(pid):
 p=pathlib.Path('/proc')/str(pid)
 try:
  st=(p/'stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'ppid':int(st[1]),'start':st[19],'state':st[0],'cmd':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
 except (FileNotFoundError,ProcessLookupError):return None
def send(p,sig):
 q=proc(p['pid'])
 if q and q['start']==p['start'] and q['state']!='Z':os.kill(p['pid'],sig);return True
 return False
server,sim=proc(4049716),proc(4076687)
if server:assert str(B) in server['cmd'] and 'setup_policy_server.py' in server['cmd'] and 'handover_block-10k-opt-v1/export-10000-ema' in server['cmd']
if sim:assert str(B/'scripts/run_eval_paired.py') in sim['cmd'] and 'flux-10k-test100' in sim['cmd']
parent=proc(sim['ppid'] if sim else server['ppid']) if (sim or server) else None
if parent:assert str(B/'scripts/launch_flux_10k_test100.sh') in parent['cmd']
allp={int(p.name):proc(int(p.name)) for p in pathlib.Path('/proc').iterdir() if p.name.isdigit()}; children=[];todo=[sim['pid']] if sim else []
while todo:
 pid=todo.pop()
 for p in allp.values():
  if p and p['ppid']==pid:children.append(p);todo.append(p['pid'])
record={'time':time.time(),'reason':'User explicitly requests stopping previous-training evaluation processes','parent':parent,'server':server,'sim':sim,'children':children}
(B/'status/old-eval-user-stop-before.json').write_text(json.dumps(record,indent=2))
if parent:send(parent,signal.SIGSTOP)
if sim:send(sim,signal.SIGTERM)
for p in children:send(p,signal.SIGINT if 'ffmpeg' in p['cmd'] else signal.SIGTERM)
if server:send(server,signal.SIGTERM)
time.sleep(3)
for p in children+[x for x in [sim,server] if x]:send(p,signal.SIGKILL)
if parent:send(parent,signal.SIGTERM);send(parent,signal.SIGCONT)
time.sleep(1)
if parent:send(parent,signal.SIGKILL)
log=(B/'status/flux-10k-test100-handover_block.log').read_text(errors='replace');log=re.sub(r'\x1b\[[0-9;]*m','',log)
scores=list(re.finditer(r'Success rate:\s*(\d+)/(\d+)\s*=>.*?current seed:\s*(\d+)',log))
record['official_count_at_stop']=list(map(int,scores[-1].groups())) if scores else None
record['after']={p['pid']:proc(p['pid']) for p in children+[x for x in [parent,sim,server] if x]}
assert not any(p and p['state']!='Z' for p in record['after'].values())
record['stopped_verified']=True;record['complete_test']=False;record['gpu_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
(B/'status/old-eval-user-cancelled.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'stopped_verified':True,'count':record['official_count_at_stop'],'gpu_after':record['gpu_after']}))
