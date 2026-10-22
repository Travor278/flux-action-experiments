from collections import Counter
from random import Random
from matched_window_sampler import MatchedWindowSampler
from flux_action.training.data import WindowDataset
from flux_action.data.windows import select_window_start

def manifest():
    episodes=[];offset=0
    for i in range(50):
        n=50+i
        episodes.append({'episode_index':i,'episode_id':str(i),'n_frames':n,'valid_ranges':[[1,n]],'from_index':offset,
                         'caption':'test','split':'train','videos':{}})
        offset+=n
    return {'episodes':episodes,'fps':15,'chunk_size':32,'total_frames':offset,'state_dim':14,'action_dim':14,'camera_order':[]}

def start_index(ds,position):
    ep=ds.episodes[position%len(ds.episodes)];visit=position//len(ds.episodes)
    key=f"{ds.seed}:{ds.epoch}:{ep['episode_index']}"
    rng=Random(key if visit==0 else f'{key}:{visit}')
    return ep['from_index']+select_window_start(ep['n_frames'],ep['valid_ranges'],rng,32)

def test_global_optimizer_batch_matches_flux_worker_sharding():
    m=manifest()
    for n in [2,4,8]:
        for micro in [1,2]:
            paired=list(MatchedWindowSampler(m))
            accum=32//(n*micro);rank_streams=[]
            for rank in range(n):
                ds=WindowDataset(m,'.','unused',seed=42,rank=rank,world_size=n,num_workers=2,windows_per_rank=micro,grad_accumulation=accum,visits_per_epoch=64)
                workers=[ds.worker_positions(w) for w in range(2)]
                batches=[]
                for b in range(ds.batches_per_rank):
                    w=b%2;k=b//2
                    batches.append([start_index(ds,p) for p in workers[w][k*micro:(k+1)*micro]])
                rank_streams.append(batches)
            for step in range(100):
                observed=[i for stream in rank_streams for batch in stream[step*accum:(step+1)*accum] for i in batch]
                assert Counter(observed)==Counter(paired[step*32:(step+1)*32])

def test_epoch_reset_reproduces_resumed_sample_sequence():
    s=MatchedWindowSampler(manifest())
    first=list(s);second=list(s);s.set_epoch(1)
    assert list(s)==second
    assert first!=second

def test_accelerate_two_rank_batches_preserve_global_window_sequence():
    import torch
    from torch.utils.data import DataLoader
    from accelerate.data_loader import prepare_data_loader
    m=manifest();streams=[]
    for rank in [0,1]:
        loader=DataLoader(range(m['total_frames']),batch_size=16,sampler=MatchedWindowSampler(m),drop_last=True)
        prepared=prepare_data_loader(loader,device=torch.device('cpu'),num_processes=2,process_index=rank,
                                     split_batches=False,rng_types=[])
        streams.append([batch.tolist() for batch in prepared])
    reference=list(MatchedWindowSampler(m))
    assert [len(s) for s in streams]==[100,100]
    for step in range(100):
        assert Counter(streams[0][step]+streams[1][step])==Counter(reference[step*32:(step+1)*32])
