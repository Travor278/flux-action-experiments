"""FLUX Trainer dataset plug with upstream rank/worker/exact-resume ordering."""
import json
from pathlib import Path
from random import Random
import numpy as np
import torch
from flux_action.training.data import WindowDataset
from flux_action.data.droid.video import decode_window
from robodojo_history_reader import RoboDojoHistoryReader, SELECTED


class RoboDojoHistoryDataset(WindowDataset):
    def __init__(self, manifest, source_root, rows_path, *, n_obs_steps, prefix_ticks=None, **kwargs):
        task = manifest['robodojo_task']
        ids = [e['episode_index'] for e in manifest['episodes']]
        if task not in SELECTED or sorted(ids) != list(SELECTED[task]):
            raise ValueError('requires exactly the selected 100 task episodes')
        if manifest['fps'] != 25 or tuple(kwargs.get('frame_hw',(480,640))) != (480,640):
            raise ValueError('native 25Hz and 480x640 required')
        if n_obs_steps < 1:
            raise ValueError('explicit history length required')
        kwargs.setdefault('frame_hw',(480,640))
        # Keep upstream topology/epoch/drop-last/worker rotation, but do not
        # apply its full-history/full-future eligibility filter to these data.
        super().__init__(manifest, source_root, rows_path, n_obs_steps=None, **kwargs)
        self.n_obs_steps = n_obs_steps
        self.prefix_ticks = prefix_ticks

    def valid_starts(self, episode_position):
        return range(self.episodes[episode_position]['n_frames'])

    def signature(self, position):
        ep = self.episodes[position % len(self.episodes)]
        visit = position // len(self.episodes)
        key = f"{self.seed}:{self.epoch}:{ep['episode_index']}"
        rng = Random(key if visit == 0 else f'{key}:{visit}')
        start = rng.randrange(ep['n_frames'])
        caption = rng.randrange(len(ep['source_episode']['tasks']))
        return start, caption, rng.getrandbits(63)

    def window(self, position):
        start, caption, seed = self.signature(position)
        return self.window_at(position % len(self.episodes), start, caption_index=caption, window_seed=seed)

    def _decode(self, path, timestamps):
        indices = np.rint(np.asarray(timestamps)*25).astype(np.int64)
        if not np.allclose(indices/25, timestamps, atol=1e-6, rtol=0):
            raise ValueError('video timestamps not on native grid')
        first, last = int(indices.min()), int(indices.max())
        rgb = decode_window(path,first,last-first+1,fps=25,frame_hw=(480,640),decoder=self.decoder,threads=1,resize=False)
        return rgb[indices-first].transpose(0,3,1,2).copy()

    def window_at(self, episode_position, start, *, caption_index=0, window_seed=0):
        ep = self.episodes[episode_position]
        first, count = ep['from_index'], ep['n_frames']
        rows = self.rows[first:first+count]
        reader = RoboDojoHistoryReader(ep['source_episode'],self.manifest['robodojo_task'],rows[:,:14],rows[:,14:],self.source_root,self.manifest['source_prefix'],self._decode)
        item = reader.window_at(start,history=self.n_obs_steps,chunk=self.chunk_size,caption_index=caption_index,prefix_ticks=self.prefix_ticks)
        item['window_seed'] = window_seed
        return {k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in item.items()}


def build_dataset(cfg, **loader_kwargs):
    """Use cfg.dataset='robodojo_history_dataset:build_dataset' with scripts on PYTHONPATH."""
    if cfg.policy.get('inference_profile') != 'history':
        raise ValueError('history policy required')
    root = Path(cfg.index_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['chunk_size'] != cfg.policy['chunk_size']:
        raise ValueError('index and policy chunk differ')
    spec = cfg.policy.get('prefix_memory_spec')
    if spec is not None:
        from prefix_memory import PrefixMemorySpec
        ticks = PrefixMemorySpec.from_dict(spec).ticks
    else:
        ticks = None
    return RoboDojoHistoryDataset(manifest,cfg.source_root,root/'rows.npy',n_obs_steps=cfg.policy['n_obs_steps'],prefix_ticks=ticks,decoder=cfg.decoder,frame_hw=cfg.frame_hw,visits_per_epoch=cfg.visits_per_epoch,**loader_kwargs)
