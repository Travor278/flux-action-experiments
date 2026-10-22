from pathlib import Path
import hashlib,json,time
p=Path('D:/Code/Project/A2R/.playwright-cli/pi05-ssh-data.tar');r={'time':time.time(),'path':str(p),'removed':False}
assert p.resolve().parent==Path('D:/Code/Project/A2R/.playwright-cli').resolve()
if p.exists():
 assert p.stat().st_size==1763051520
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='9de1b340ea4cbecb9307b158711f7394adafc2c5c327e4ca96ce6bd5ad9c7ee9'
 p.unlink();r['removed']=True;r['bytes']=1763051520
(Path(__file__).parent/'old-browser-transfer-cleanup.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
