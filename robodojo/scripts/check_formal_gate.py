from pathlib import Path
import hashlib,json,shutil,sys
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
task=sys.argv[1];assert task in ('cover_blocks','match_and_pick_from_conveyor')
gate=json.loads((R/'status/formal-ready.json').read_text());assert gate['ready']
for name,digest in gate['files'].items():
    path=(R/name).resolve();assert path.is_relative_to(R.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
assert json.loads((R/'status/full-preflight-complete.json').read_text())['complete']
assert json.loads((R/'status/full-preflight-export-audit.json').read_text())['verified']
assert json.loads((R/'status/export-inference-current.json').read_text())['verified']
scene=json.loads((R/'status/scene-control-current.json').read_text())
assert scene['render_passed'] and all(scene['tasks'][n]['verified'] and scene['tasks'][n]['control_ticks']==8 for n in ('cover_blocks','match_and_pick_from_conveyor'))
assert json.loads((R/'status/selected-download-independent-audit.json').read_text())['verified']
from flux_action.training.trainer import TrainConfig
from robodojo_training_contract import custom_data_contract
cfg=TrainConfig.from_file(R/f'configs/{task}-10k-prefix-v1.json')
assert cfg.steps==10000 and cfg.global_batch(2)==32 and cfg.keep_checkpoints==1 and cfg.keep_every is None
assert custom_data_contract(cfg)==gate['data_contracts'][task]
out=Path(cfg.output_dir);assert out==Path(str(R).replace('/hdd/','/ssd/'))/'runs'/f'{task}-10k-prefix-v1'
assert shutil.disk_usage(out.parent if out.parent.exists() else out.parents[1]).free>350_000_000_000
print('ROBODOJO_FORMAL_GATE_PASS',task,flush=True)
