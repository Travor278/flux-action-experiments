"""Uniform episode visits and ALL real observation ticks; padding never crosses episodes.

Only absolute native commands and one-observation policies are supported. Padding repeats
the last recorded value for shape stability; temporal validity, not zero values, defines loss.
"""
from random import Random
from pathlib import Path
import numpy as np
import torch
from flux_action.training.data import WindowDataset, load_manifest
from flux_action.data.droid.episodes import select_caption
from flux_action.data.droid.video import decode_window


class TailWindows(WindowDataset):
    def valid_starts(self, episode_position):
        return list(range(self.episodes[episode_position]['n_frames']))

    def window(self, position):
        pos = position % len(self.episodes)
        ep = self.episodes[pos]
        visit = position // len(self.episodes)
        rng = Random(f"tail-v1:{self.seed}:{self.epoch}:{ep['episode_index']}:{visit}")
        start = rng.randrange(ep['n_frames'])
        return self.window_at(pos, start, task=select_caption(ep['caption'], rng),
                              window_seed=rng.getrandbits(63))

    def window_at(self, episode_position, start, *, task=None, window_seed=0):
        ep = self.episodes[episode_position]
        n, k = ep['n_frames'], self.chunk_size
        assert 0 <= start < n and self.n_obs_steps is None
        offset = ep['from_index']
        # Clamp indices INSIDE this episode before adding the global row offset.
        idx = np.minimum(start + np.arange(k + 1), n - 1)
        rows = np.array(self.rows[offset + idx], copy=True)
        assert np.isfinite(rows).all()
        s = self.state_dim
        item = {'state': torch.from_numpy(rows[0, :s].copy()),
                'action': torch.from_numpy(rows[:k, s:].copy()),
                'action_time_mask': torch.arange(k) < min(k, n-start),
                'video_valid_frames': min(k+1, n-start),
                'task': task if task is not None else select_caption(ep['caption']),
                'episode_index': ep['episode_index'], 'start': start,
                'window_seed': window_seed}
        # At reset the measured/recorded state is the only preceding hold command.
        # This field is ignored by the required absolute action parameterization.
        previous = np.array(self.rows[offset+start-1, s:] if start else rows[0, :s], copy=True)
        item['action_prev'] = torch.from_numpy(previous)
        count = min(k+1, n-start)
        for camera in self.cameras:
            video = ep['videos'][camera]
            assert video is not None, 'RoboTwin repair requires all three real cameras'
            frames = decode_window(str(Path(self.source_root)/video['file']),
                                   video['first_frame']+start, count, fps=self.fps,
                                   frame_hw=self.frame_hw, decoder=self.decoder,
                                   resize=self.resize[camera])
            assert len(frames) == count
            frames = np.concatenate([frames, np.repeat(frames[-1:], k+1-count, axis=0)], axis=0)
            item[f'images.{camera}'] = torch.from_numpy(frames).permute(0,3,1,2).contiguous()
        return item


def build(config, **kwargs):
    assert config.policy.get('action_parameterization', 'absolute') == 'absolute'
    assert config.policy.get('n_obs_steps', 1) == 1
    index = Path(config.index_dir)
    manifest = load_manifest(index/'manifest.json')
    assert manifest['codebase_version'] == 'v3.0'
    assert len(manifest['episodes']) == 50
    return TailWindows(manifest, config.source_root, index/'rows.f32.npy',
                       decoder=config.decoder, frame_hw=config.frame_hw,
                       visits_per_epoch=config.visits_per_epoch, **kwargs)
