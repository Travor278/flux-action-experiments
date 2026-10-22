import json
import unittest
import torch
from prefix_memory import PrefixMemoryBank, PrefixMemorySpec, prefix_relative_seconds


class PrefixMemoryTests(unittest.TestCase):
    def test_offline_online_parity_on_every_tick_and_reset(self):
        spec = PrefixMemorySpec()
        bank = PrefixMemoryBank(['images.head','images.wrist'],spec)
        for episode in range(2):
            bank.reset()
            for tick in range(101):
                # No reliance on inference cadence: all queued ticks are appended.
                images = {k:torch.full((1,3,2,3),tick+episode*110,dtype=torch.uint8) for k in bank.keys}
                out = bank.append(images)
                indices = spec.indices(tick)
                self.assertEqual(out['prefix_tick_indices'].tolist(),[list(indices)])
                self.assertEqual(out['prefix.images.head'][0,:,0,0,0].tolist(),[t+episode*110 for t in indices])
                self.assertTrue((prefix_relative_seconds(out['prefix_tick_indices'],out['episode_tick'],spec)<=0).all())
                images['images.head'].zero_() # retained snapshots must own their storage
            self.assertEqual(sorted(bank.anchors),[0,25,50])
            self.assertEqual(bank.anchors[0]['images.head'][0,0,0,0].item(),episode*110)

    def test_future_and_missing_ticks_rejected(self):
        spec = PrefixMemorySpec()
        with self.assertRaises(ValueError):
            prefix_relative_seconds(torch.tensor([[0,25,50]]),torch.tensor([10]),spec)
        with self.assertRaises(ValueError):
            PrefixMemorySpec((0,25,25))

    def test_serialization_preserves_causal_contract(self):
        spec = PrefixMemorySpec()
        restored = PrefixMemorySpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        self.assertEqual(restored,spec)
        data = spec.to_dict();data['unavailable_rule']='future_frame'
        with self.assertRaises(ValueError): PrefixMemorySpec.from_dict(data)


if __name__ == '__main__': unittest.main()
