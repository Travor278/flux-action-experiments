"""Prepare reviewable configurations only; cannot submit or create run outputs."""
from pathlib import Path
import copy,json,time,hashlib
from flux_action.training.trainer import TrainConfig
from flux_action.config import PolicyConfig
from robodojo_training_contract import custom_data_contract
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
S=Path(str(R).replace('/hdd/','/ssd/'))
assert json.loads((R/'status/full-preflight-complete.json').read_text())['complete']
assert json.loads((R/'status/selected-download-independent-audit.json').read_text())['verified']
assert json.loads((R/'status/export-inference-current.json').read_text())['verified']
base=json.loads((R/'configs/full-preflight-v1.json').read_text())
stats=json.loads((R/'status/numeric-data-verified.json').read_text())['tasks']
records={}
for task in ['cover_blocks','match_and_pick_from_conveyor']:
    cfg=copy.deepcopy(base)
    cfg.update(index_dir=str(R/'assets/history-index-preflight-v1'/task),output_dir=str(S/'runs'/f'{task}-10k-prefix-v1'),steps=10000,
               frozen_steps=200,trunk_warmup_steps=600,heads_warmup_steps=200,
               cooldown_start=9000,cooldown_steps=1000,windows_per_rank=2,grad_accumulation=8,
               prefetch_factor=16,checkpoint_every=1000,keep_checkpoints=1,keep_every=None,log_every=10,
               resume='auto',reseed_on_resume=False)
    cfg['policy'].update(action_normalization={'q01':stats[task]['action_q01'],'q99':stats[task]['action_q99']},
                         state_normalization={'q01':stats[task]['state_q01'],'q99':stats[task]['state_q99']})
    assert not Path(cfg['output_dir']).exists()
    tc=TrainConfig(**cfg);pc=PolicyConfig(**cfg['policy']);assert tc.global_batch(2)==32
    assert pc.n_obs_steps==8 and pc.chunk_size==32 and pc.fps==25
    path=R/'configs'/f'{task}-formal-draft.json'
    content=json.dumps(cfg,indent=2)
    if path.exists():assert path.read_text()==content,'Refuse overwriting a changed draft'
    else:path.write_text(content)
    records[task]={'config':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'data_contract':custom_data_contract(tc),'windows_per_update':32,'planned_updates':10000}
result={'time':time.time(),'drafts_prepared':True,'ready_for_submission':False,'records':records,
        'decision':'Initial10k budget analogous to prior single-task runs, not an official RoboDojo standard. Micro2 preserves proven memory fit; prefetch16 needs steady throughput check. No formal output/job created.',
        'remaining':['Actual scene-control check job09f454 currently queued','Steady full-update throughput/prefetch measurement','Immutable launch gate and early100-step save/safe-stop runner']}
(R/'status/formal-drafts.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
