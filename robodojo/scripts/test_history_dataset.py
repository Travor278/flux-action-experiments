import unittest
from robodojo_history_dataset import RoboDojoHistoryDataset
from flux_action.training.data import build_dataloader


def manifest():
    return {'robodojo_task':'cover_blocks','fps':25,'chunk_size':32,'total_frames':500,'state_dim':14,'action_dim':14,'camera_order':['head','left_wrist','right_wrist'],'action_prev':True,'episodes':[{'episode_index':i,'n_frames':5,'source_episode':{'tasks':['one','two']}} for i in range(300,400)]}


class SignatureDataset(RoboDojoHistoryDataset):
    def window(self, position):
        start, caption, seed = self.signature(position)
        return {'task':'test','position':position,'start':start,'caption':caption,'window_seed':seed}


def dataset(rank=0, skip=0):
    return SignatureDataset(manifest(),'.','unused.npy',n_obs_steps=64,seed=42,epoch=3,rank=rank,world_size=2,num_workers=2,windows_per_rank=4,grad_accumulation=4,visits_per_epoch=2,skip_batches=skip)


def collect(ds):
    return [{k:v.tolist() for k,v in b.items() if k!='task'} for b in build_dataloader(ds,prefetch_factor=2,pin_memory=False)]


class DatasetTests(unittest.TestCase):
    def test_real_two_worker_prefetch_resume_suffix(self):
        full = collect(dataset())
        for skip in (1,4):
            self.assertEqual(collect(dataset(skip=skip)),full[skip:])

    def test_rank_positions_do_not_overlap_and_all_frames_eligible(self):
        a,b = dataset(0),dataset(1)
        pa = {p for w in range(2) for p in a.worker_positions(w)}
        pb = {p for w in range(2) for p in b.worker_positions(w)}
        self.assertFalse(pa & pb)
        self.assertEqual(len(pa),len(pb))
        self.assertEqual(a.episodes_without_history,0)
        self.assertEqual(list(a.valid_starts(0)),list(range(5)))
        self.assertEqual(a.signature(14),b.signature(14))


if __name__ == '__main__':
    unittest.main()
