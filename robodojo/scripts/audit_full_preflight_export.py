from pathlib import Path
import hashlib,json,pickle,time
import torch
from safetensors import safe_open
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=Path(str(R).replace('/hdd/','/ssd/'))/'runs/full-preflight-v1'
torch.set_num_threads(4)
complete=json.loads((R/'status/full-preflight-complete.json').read_text());assert complete['complete'] and complete['exact_fingerprint_restore_step']==3
for rank in (0,1):
    proof=json.loads((out/f'exact-resume-step3-rank{rank}.json').read_text())
    expected=json.loads((out/f'exact-state-step3-rank{rank}.json').read_text())
    assert proof['verified'] and proof['step']==3 and proof['fingerprints']==expected
p=out/'step-4';assert (p/'COMPLETE').exists()
state=json.loads((p/'state.json').read_text());assert state['step']==4 and state['world_size']==2
components={}
for name in ['model','optimizer','ema_0p10','ema_0p05']:
    with (p/name/'.metadata').open('rb') as f:meta=pickle.load(f)
    for v in meta.storage_data.values():
        file=(p/name/v.relative_path).resolve();assert file.is_relative_to((p/name).resolve())
        assert v.offset>=0 and v.length>0 and v.offset+v.length<=file.stat().st_size
    components[name]=len(meta.state_dict_metadata)
export=out/'export-4-ema';manifest=json.loads((export/'manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    with (export/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
config=json.loads((export/'config.json').read_text());assert config['prefix_memory_spec']['version']=='causal_prefix_rgb_v1' and config['action_dim']==14 and config['fps']==25
with safe_open(export/'model.safetensors',framework='pt',device='cpu') as f:
    names=list(f.keys());assert len(names)==451
    for name in names:assert torch.isfinite(f.get_tensor(name)).all(),name
result={'time':time.time(),'verified':True,'exact_restore_step':3,'both_rank_fingerprints_match':True,'step4_components':components,'export':str(export),'export_sha256':manifest['sha256'],'export_tensors':451,'all_export_values_finite':True,'inference_verified':False,'scope':'Independent completed diagnostic checkpoint/export audit; retain checkpoint until actual export inference succeeds'}
(R/'status/full-preflight-export-audit.json').write_text(json.dumps(result,indent=2))
print('FULL_PREFLIGHT_EXPORT_INDEPENDENT_PASS',flush=True)
