from pathlib import Path
import json,time
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
p=S/'runs/click_bell';t=(p/'train.log').read_text(errors='replace');lines=t.splitlines();idx=[i for i,l in enumerate(lines) if 'Traceback' in l or 'Error' in l or 'error' in l];a={'time':time.time(),'errors':['\n'.join(lines[max(0,i-3):i+38]) for i in idx[:8]],'tail':t[-12000:],'checkpoint_files':[{'path':str(f.relative_to(p)),'bytes':f.stat().st_size} for f in p.glob('step-*/*/*') if f.is_file()]};(B/'status/click-checkpoint-failure.json').write_text(json.dumps(a,indent=2))
