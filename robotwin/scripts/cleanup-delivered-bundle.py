from pathlib import Path
import json,time
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008').resolve()
p=B/'assets/pi05-ssh-data.tar'
assert p.resolve().is_relative_to(B) and not p.is_symlink()
if p.exists():
 size=p.stat().st_size;assert size==1763051520;p.unlink()
 (B/'status/pi05-ssh-source-cleanup.json').write_text(json.dumps({'time':time.time(),'removed':str(p),'bytes':size,'destination_verified':True,'destination_record':'/home/USER/codex_remote_ops/flux_robotwin_pi05_20261008/status/ssh-transfer-verified.json'},indent=2))
