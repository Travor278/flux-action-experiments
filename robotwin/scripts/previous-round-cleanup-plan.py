from pathlib import Path
import json
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008').resolve()
S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008').resolve()
items=[]
for p in [S/'runs/click_bell',S/'runs/handover_block',B/'runs/invalid-profile-shadowing-20261008',B/'runs/checkpoint-cpu-smoke']:
 if not p.exists():continue
 assert not p.is_symlink() and (p.resolve().is_relative_to(B) or p.resolve().is_relative_to(S))
 files=[f for f in p.rglob('*') if f.is_file() and not f.is_symlink()]
 items.append({'path':str(p),'files':len(files),'bytes':sum(f.stat().st_size for f in files),'allocated_bytes':sum(f.stat().st_blocks*512 for f in files),'top_files':[(str(f.relative_to(p)),f.stat().st_size) for f in sorted(files,key=lambda f:f.stat().st_size,reverse=True)[:8]],'complete_markers':[str(f) for f in files if f.name=='COMPLETE']})
(B/'status/previous-round-cleanup-plan.json').write_text(json.dumps(items,indent=2))
