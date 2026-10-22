from pathlib import Path
import json
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008')
lines=(S/'runs/click_bell/train.log').read_text(errors='replace').splitlines();out=[]
for i,l in enumerate(lines):
 if 'WARN' in l or 'out of memory' in l.lower():out.append('\n'.join(lines[max(0,i-2):i+5]))
(B/'status/checkpoint-causal-warnings.json').write_text(json.dumps(out,indent=2))
