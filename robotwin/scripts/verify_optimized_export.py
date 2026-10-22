import pathlib,json,hashlib,sys,time
from safetensors import safe_open
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
task,step=sys.argv[1],int(sys.argv[2]);assert task in ['click_bell','handover_block'] and step in [1000,3000,5000,10000]
R=pathlib.Path(str(B).replace('/hdd/','/ssd/'))/f'runs/{task}-10k-opt-v1'
E=R/f'export-{step}-ema';M=json.loads((E/'manifest.json').read_text())
assert (R/f'step-{step}/COMPLETE').is_file()
state=json.loads((R/f'step-{step}/state.json').read_text());assert state['step']==step and state['world_size']==2
for name,sha in M['sha256'].items():
 with (E/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==sha
with safe_open(str(E/'model.safetensors'),framework='pt',device='cpu') as f: assert len(f.keys())==451
record={'time':time.time(),'verified':True,'task':task,'step':step,'export':str(E),'manifest':M}
(R/f'export-{step}-verified.json').write_text(json.dumps(record,indent=2))
if step==10000:(R/'TRAIN_COMPLETE.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'task':task,'step':step,'verified':True}),flush=True)
