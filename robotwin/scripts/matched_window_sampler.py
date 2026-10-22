"""Use the same episode visits and window-start RNG as the pinned FLUX loader."""
from random import Random
from torch.utils.data import Sampler
from flux_action.data.windows import select_window_start
from flux_action.training.data import epoch_order

class MatchedWindowSampler(Sampler):
    def __init__(self, manifest, seed=42, visits=64, initial_epoch=0):
        self.episodes=[e for e in manifest['episodes'] if e.get('split','train')=='train']
        self.chunk_size=manifest['chunk_size']
        self.seed=seed;self.visits=visits;self.epoch=initial_epoch

    def __len__(self):return len(self.episodes)*self.visits

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def __iter__(self):
        epoch=self.epoch;self.epoch+=1
        for position in epoch_order(len(self),self.seed,epoch):
            episode=self.episodes[position%len(self.episodes)]
            visit=position//len(self.episodes)
            key=f"{self.seed}:{epoch}:{episode['episode_index']}"
            rng=Random(key if visit==0 else f'{key}:{visit}')
            start=select_window_start(episode['n_frames'],episode['valid_ranges'],rng,self.chunk_size)
            if start is None:raise ValueError('No complete window in paired training episode')
            yield episode['from_index']+start
