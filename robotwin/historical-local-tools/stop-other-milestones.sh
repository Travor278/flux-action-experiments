set -euo pipefail
python3 - <<'PY'
import pathlib,json,subprocess,time,os,signal
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');root=int((B/'status/flux-milestone-validation.pid').read_text().strip());proc=pathlib.Path('/proc')
def identity(pid):
 try:return (proc/str(pid)/'stat').read_text().split(') ')[1].split()[19]
 except FileNotFoundError:return None
cmd=(proc/str(root)/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert str(B/'scripts/launch_flux_milestone_validation_v2.sh') in cmd,cmd
root_id=identity(root);os.kill(root,signal.SIGSTOP)
rows={}
for line in subprocess.check_output(['ps','-eo','pid,ppid,args','--no-headers'],text=True).splitlines():
 p,pp,c=line.strip().split(None,2);rows[int(p)]={'pid':int(p),'ppid':int(pp),'cmd':c}
own={root};depth={root:0}
while True:
 added={p for p,r in rows.items() if r['ppid'] in own and p not in own}
 if not added:break
 for p in added:depth[p]=depth[rows[p]['ppid']]+1
 own|=added
assert len(own)<120 and os.getpid() not in own
ids={p:identity(p) for p in own};events=[]
for p in sorted(own-{root},key=lambda x:depth[x],reverse=True):
 if ids[p] is not None and identity(p)==ids[p]:
  try:os.kill(p,signal.SIGTERM);events.append({'pid':p,'signal':'TERM'})
  except ProcessLookupError:pass
if identity(root)==root_id:
 os.kill(root,signal.SIGTERM);os.kill(root,signal.SIGCONT)
for _ in range(20):
 alive=[p for p in own if ids[p] is not None and identity(p)==ids[p]]
 if not alive:break
 time.sleep(.5)
for p in alive:
 if identity(p)==ids[p]:
  try:os.kill(p,signal.SIGKILL);events.append({'pid':p,'signal':'KILL'})
  except ProcessLookupError:pass
record={'time':time.time(),'reason':'User restricts evaluation to 10000-step models only','root':root,'processes':[rows[p] for p in sorted(own)],'signals':events,'cancelled_other_milestones':True}
(B/'status/flux-milestone-user-cancelled.json').write_text(json.dumps(record,indent=2))
PY
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
flock -n "$B/status/flux-milestone-validation.lock" -c 'true'
printf 'LOCK_RELEASED\n' > "$B/status/flux-milestone-stop-verified.txt"
