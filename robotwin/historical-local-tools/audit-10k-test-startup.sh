set -euo pipefail
python3 - <<'PY'
import pathlib,json,time,re,hashlib,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));E=B/'evaluation/click_bell/flux-10k-test100';p=json.loads((E/'protocol.json').read_text())
assert p['test_num']==100 and p['seed']==0 and p['task_config']=='demo_clean' and p['instruction_type']=='seen' and p['env_cfg_type']=='aloha_agilex' and p['action_type']=='joint' and p['expert_check'] is True and p['frequency']==15
out={'time':time.time(),'protocol_verified':True,'protocol':p,'debug':{},'disk_free_bytes':shutil.disk_usage(S).free,'transfer_remnants':[]}
for encoded in ['false','true']:
 f=B/f'status/flux-10k-test100-click_bell-debug-{encoded}.log';s=f.read_text(errors='replace');assert len(s)>0 and 'Traceback' not in s
 out['debug'][encoded]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'tail':s[-1600:]}
for root in [B/'assets',S/'hf-publish',S/'hf-transfer-cache']:
 if root.exists():
  for f in root.glob('*'):
   if f.is_file() and f.name.endswith(('.partial','.part','.tmp','.tar')):out['transfer_remnants'].append(str(f))
(B/'status/flux-10k-test-startup-audit.json').write_text(json.dumps(out,indent=2))
PY
