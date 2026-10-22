from pathlib import Path
import hashlib,json,tempfile,time,types
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
p=R/'flux-action/src/flux_action/training/trainer.py'
receipt=R/'status/prefix-resume-contract-verified.json'
if receipt.exists():
    old=json.loads(receipt.read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==old['after_sha256']
    print('ALREADY_VERIFIED');raise SystemExit(0)
before=p.read_bytes();text=before.decode()
needle='            "history_snapshots",\n            "condition_on_past_actions",'
assert text.count(needle)==1 and '"prefix_memory_spec"' not in text[text.index('def _validate_resume_contract'):text.index('def _resume(')]
(R/'status/trainer-before-prefix-resume-guard.py').write_bytes(before)
updated=text.replace(needle,'            "history_snapshots",\n            "prefix_memory_spec",\n            "condition_on_past_actions",')
temp=p.with_suffix('.py.tmp');temp.write_text(updated);temp.replace(p)
from flux_action.training.trainer import Trainer
from flux_action.config import PolicyConfig
export=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1/export-4-ema'
cfg=json.loads((export/'config.json').read_text());config=PolicyConfig(**cfg)
fake=types.SimpleNamespace(policy=types.SimpleNamespace(config=config),_data_contract=lambda:{},config=types.SimpleNamespace(to_dict=lambda:{}))
checks=[]
with tempfile.TemporaryDirectory(prefix='resume-contract-',dir=R/'status') as td:
    root=Path(td);(root/'state.json').write_text('{"extra":{}}')
    (root/'config.json').write_text(json.dumps(cfg))
    Trainer._validate_resume_contract(fake,root);checks.append('identical real exported config accepted')
    for label,spec in [('changed',dict(cfg['prefix_memory_spec'],ticks=[0,20,50])),('removed',None)]:
        altered=dict(cfg,prefix_memory_spec=spec)
        (root/'config.json').write_text(json.dumps(altered))
        try:Trainer._validate_resume_contract(fake,root)
        except ValueError as exc:
            assert 'prefix_memory_spec' in str(exc),str(exc);checks.append(label+' prefix rejected')
        else:raise AssertionError(label+' prefix unexpectedly accepted')
result={'time':time.time(),'verified':True,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checks':checks,'scope':'Actual Trainer resume contract exercised with real diagnostic exported config; no model allocation or checkpoint recreation.'}
receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
