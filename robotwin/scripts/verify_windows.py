import json,pathlib
import numpy as np
from flux_action.training.data import DroidWindowDataset
from flux_action.training.trainer import TrainConfig
from flux_action.config import PolicyConfig
B=pathlib.Path(__file__).resolve().parents[1]
out={}
for task in ['click_bell','handover_block']:
    cfg=TrainConfig.from_file(B/'configs'/f'{task}.json')
    PolicyConfig(**cfg.policy).validate_training()
    index=B/'assets/index'/task
    manifest=json.loads((index/'manifest.json').read_text())
    ds=DroidWindowDataset(manifest,B/'assets/lerobot'/task,index/'rows.f32.npy',seed=42,decoder='pyav',frame_hw=(240,320),visits_per_epoch=64)
    rows=np.load(index/'rows.f32.npy',mmap_mode='r')
    checked=[]
    for position in [0,25,49]:
        w=ds.window(position)
        assert w['state'].shape==(14,) and w['action'].shape==(32,14)
        assert all(tuple(w[k].shape)==(33,3,240,320) for k in cfg.policy['camera_keys'])
        assert np.isfinite(w['state'].numpy()).all() and np.isfinite(w['action'].numpy()).all()
        first=manifest['episodes'][position]['from_index']+w['start']
        np.testing.assert_array_equal(w['state'].numpy(),rows[first,:14])
        np.testing.assert_array_equal(w['action'].numpy(),rows[first:first+32,14:])
        checked.append({'episode':int(w['episode_index']),'start':int(w['start']),'task':w['task']})
    out[task]={'frames':manifest['total_frames'],'counts':manifest['counts'],'windows_checked':checked,'config_valid':True}
(B/'status/windows-verified.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
