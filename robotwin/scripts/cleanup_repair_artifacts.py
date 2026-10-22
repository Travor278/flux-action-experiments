from pathlib import Path
import json,time,shutil
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008').resolve();S=Path('/workspace/ssd/codex_remote_ops/flux_piperx_20261008').resolve()
m=json.loads((B/'status/flux-final-artifacts-verified.json').read_text());assert all(m['tasks'][t]['verified'] for t in ['click_bell','handover_block'])
p=S/'runs/checkpoint-repair-v1';assert p.resolve().is_relative_to(S/'runs') and not p.is_symlink()
if p.exists():
 assert (p/'step-3/COMPLETE').exists()
 files=[x for x in p.rglob('*') if x.is_file() and not x.is_symlink()]
 receipt={'time':time.time(),'retired_path':str(p),'files':len(files),'bytes':sum(x.stat().st_size for x in files),'reason':'Production step3000 checkpoints and both exports verified; repair-only model no longer used','state':json.loads((p/'step-3/state.json').read_text()),'export_manifest':json.loads((p/'export-3-ema/manifest.json').read_text())}
 (B/'status/repair-artifact-cleanup.json').write_text(json.dumps(receipt,indent=2));shutil.rmtree(p)
 print(json.dumps({k:v for k,v in receipt.items() if k not in ['state','export_manifest']}))
