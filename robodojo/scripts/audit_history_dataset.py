"""Validate Trainer's dataset plug against separately decoded source-window receipts."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import torch
from flux_action.training.trainer import _import_dataset_builder

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
reference = json.loads((R/'status/history-reader-cpu-audit.json').read_text())
index_audit = json.loads((R/'status/history-preflight-index-verified.json').read_text())
build = _import_dataset_builder('robodojo_history_dataset:build_dataset')
out = {'time':time.time(),'scope':'Real selected compact rows, Trainer dataset plug and contiguous RGB decoder; full Trainer GPU save/resume still pending','windows':[]}
datasets = {}
for task in index_audit['tasks']:
    root = R/'assets/history-index-preflight-v1'/task
    for filename, field in [('rows.npy','rows_sha256'),('manifest.json','manifest_sha256')]:
        with (root/filename).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest() == index_audit['tasks'][task][field]
    cfg = SimpleNamespace(policy={'inference_profile':'history','n_obs_steps':8,'chunk_size':32},index_dir=str(root),source_root=str(R/'official-source'),decoder='pyav',frame_hw=(480,640),visits_per_epoch=2)
    datasets[task] = build(cfg,seed=42,epoch=0,rank=0,world_size=2,num_workers=2,windows_per_rank=4,skip_batches=0,grad_accumulation=4)
for case in reference['decoded_windows']:
    ds = datasets[case['task']]
    position = next(i for i,e in enumerate(ds.episodes) if e['episode_index'] == case['episode'])
    item = ds.window_at(position,case['start'])
    assert item['action_time_mask'].sum().item() == case['valid_actions']
    assert item['video_valid_frames'].sum().item() == case['valid_video_frames']
    assert item['state'].shape == (8,14) and item['action'].shape == (32,14)
    for key, digest in case['rgb'].items():
        assert isinstance(item[key],torch.Tensor)
        assert hashlib.sha256(item[key].numpy().tobytes()).hexdigest() == digest, (case,key)
    out['windows'].append({k:v for k,v in case.items() if k!='rgb'})
    del item
out['verified'] = len(out['windows']) == len(reference['decoded_windows'])
out['finished'] = time.time()
(R/'status/history-dataset-real-audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out),flush=True)
