from pathlib import Path
import hashlib,json,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
proof=json.loads((R/'status/selected-download-verified.json').read_text())
assert proof['complete'] and len(proof['files'])==15400
assert (R/'RoboDojo/Assets').resolve()==(R/'official-source/Assets').resolve()
assert 'SCENE_SCRIPTS_SYNTAX_PASS' in (R/'status/scene-control-syntax.log').read_text()
names=['probe_scene_control.py','launch_scene_control.sh']
record={'time':time.time(),'ready':True,'files':{n:hashlib.sha256((R/'scripts'/n).read_bytes()).hexdigest() for n in names},'asset_receipt_sha256':hashlib.sha256((R/'status/selected-download-verified.json').read_bytes()).hexdigest(),'scope':'Two tasks each8 held-state controls on actual official scene, threeRGB/state/prompt/adapterpacking. No model selection, no expert-replay claim and no benchmark score.'}
(R/'status/scene-control-ready.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
