"""Check Trainer plug prefix pixels against independent all-episode anchor audit."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
from prefix_memory import PrefixMemorySpec
from robodojo_history_dataset import build_dataset

R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
coverage=json.loads((R/'status/prefix-anchor-coverage.json').read_text())
assert coverage['complete']
reference={e['episode']:e for e in coverage['episodes']}
spec=PrefixMemorySpec()
out={'time':time.time(),'scope':'Actual selected source images through Trainer dataset plug; not real VAE/full-model GPU proof','cases':[]}
for task,ids in [('cover_blocks',[300,324]),('match_and_pick_from_conveyor',[1600,1625])]:
    cfg=SimpleNamespace(index_dir=str(R/'assets/history-index-preflight-v1'/task),source_root=str(R/'official-source'),policy={'inference_profile':'history','n_obs_steps':8,'chunk_size':32,'prefix_memory_spec':spec.to_dict()},decoder='pyav',frame_hw=(480,640),visits_per_epoch=2)
    ds=build_dataset(cfg,seed=42,rank=0,world_size=2,num_workers=2,windows_per_rank=4,grad_accumulation=4)
    for episode in ids:
        index=next(i for i,e in enumerate(ds.episodes) if e['episode_index']==episode)
        for tick in [0,ds.episodes[index]['n_frames']-1]:
            item=ds.window_at(index,tick)
            assert item['prefix_tick_indices'].tolist()==list(spec.indices(tick))
            assert item['episode_tick']==tick
            for camera,key in [('cam_high','images.head'),('cam_left_wrist','images.left_wrist'),('cam_right_wrist','images.right_wrist')]:
                assert item['prefix.'+key+'_valid_mask'].all()
                for a,source_tick in enumerate(spec.indices(tick)):
                    expected=reference[episode]['cameras'][camera]['rgb_sha256'][spec.ticks.index(source_tick)]
                    assert hashlib.sha256(item['prefix.'+key][a].numpy().transpose(1,2,0).tobytes()).hexdigest()==expected
            out['cases'].append({'task':task,'episode':episode,'tick':tick,'prefix_indices':item['prefix_tick_indices'].tolist(),'valid_actions':int(item['action_time_mask'].sum())})
            del item
out['verified']=len(out['cases'])==8
out['finished']=time.time()
(R/'status/prefix-dataset-real-audit.json').write_text(json.dumps(out,indent=2))
print('PREFIX_TRAINER_DATASET_REAL_RGB_PASS',flush=True)
