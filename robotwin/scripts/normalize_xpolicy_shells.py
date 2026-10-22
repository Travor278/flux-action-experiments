from pathlib import Path
import json,subprocess,time,hashlib
B=Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
paths=list((B/'XPolicyLab/utils').glob('*.sh'))+list((B/'XPolicyLab/policy/FLUX_Action').glob('*.sh'))
changed=[]
for p in paths:
 raw=p.read_bytes()
 if b'\r\n' in raw:
  fixed=raw.replace(b'\r\n',b'\n');tmp=p.with_suffix(p.suffix+'.lf-tmp');tmp.write_bytes(fixed);tmp.chmod(p.stat().st_mode);tmp.replace(p);changed.append(str(p.relative_to(B)))
 subprocess.run(['bash','-n',str(p)],check=True)
(B/'status/xpolicy-shell-lf-fix.json').write_text(json.dumps({'time':time.time(),'changed':changed,'bash_n_pass':len(paths)},indent=2))
