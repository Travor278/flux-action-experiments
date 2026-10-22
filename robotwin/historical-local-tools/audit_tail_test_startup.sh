B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import json,pathlib,time,hashlib,re,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');E=B/'evaluation/click_bell/flux-tail-10k-test100';P=json.loads((E/'protocol.json').read_text());r={'time':time.time(),'task':'click_bell','protocol':P,'rpc':{}}
assert P['test_num']==100 and P['seed']==0 and P['task_config']=='demo_clean' and P['instruction_type']=='seen' and P['expert_check'] and P['frequency']==15
assert P['ckpt_setting']=='flux-tail-10k-test100'
for flag in ['false','true']:
 p=B/f'status/flux-tail-10k-test100-click_bell-debug-{flag}.log';s=p.read_text(errors='replace');assert 'Traceback' not in s and len(s)>0
 r['rpc'][flag]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tail':s[-650:]}
cmd=pathlib.Path('/proc/1781375/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'click_bell-10k-tail-v1/export-10000-ema' in cmd;r['server_cmdline']=cmd
plan=json.loads((B/'status/flux-tail-final-test-plan.json').read_text());assert 'click_bell-10k-tail-v1' in plan['tasks']['click_bell']['checkpoint']
log=re.sub(r'\x1b\[[0-9;]*m','',(B/'status/flux-tail-10k-test100-click_bell.log').read_text(errors='replace'));scores=re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>.*?current seed:\s*(\d+)',log)
assert all(int(x[1])==i for i,x in enumerate(scores,1));assert len(set(x[2] for x in scores))==len(scores)
r['count_at_audit']=list(map(int,scores[-1]));r['complete_test']=False;r['protocol_and_current_checkpoint_verified']=True
(B/'status/flux-tail-test-startup-audit.json').write_text(json.dumps(r,indent=2))
PY
