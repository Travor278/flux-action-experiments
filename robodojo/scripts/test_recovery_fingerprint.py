import unittest
from types import SimpleNamespace
import torch
from train_full_preflight import recovery_fingerprint
from flux_action.training.ema import PowerEMA
from flux_action.training.data import DataPosition


class RecoveryFingerprintTests(unittest.TestCase):
    def test_real_checkpoint_rng_api_and_component_sensitivity(self):
        torch.manual_seed(4)
        policy=torch.nn.Module();policy.dit=torch.nn.Linear(4,3)
        opt=torch.optim.AdamW(policy.parameters(),lr=.01)
        policy.dit(torch.ones(2,4)).square().mean().backward();opt.step();opt.zero_grad()
        ema=PowerEMA.from_module(policy,.1)
        t=SimpleNamespace(policy=policy,optimizer=opt,emas={'ema':ema},position=DataPosition())
        first=recovery_fingerprint(t)
        self.assertEqual(first,recovery_fingerprint(t))
        with torch.no_grad():policy.dit.weight[0,0].add_(1)
        changed=recovery_fingerprint(t)
        self.assertNotEqual(first['model'],changed['model'])
        self.assertEqual(first['optimizer'],changed['optimizer'])
        self.assertEqual(first['emas'],changed['emas'])
        torch.rand(1)
        self.assertNotEqual(changed['rng'],recovery_fingerprint(t)['rng'])


if __name__=='__main__':unittest.main()
