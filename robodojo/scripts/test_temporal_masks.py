import unittest
from types import SimpleNamespace
import torch
from flux_action.processing import packing, temporal_masking as tm


class TemporalMaskTests(unittest.TestCase):
    def test_all_valid_bitwise_equivalent_to_upstream(self):
        torch.manual_seed(42)
        p = {'x_video':torch.randn(1,12,5), 'x_action':torch.randn(1,8,3)}
        t = {k:torch.randn_like(v) for k,v in p.items()}
        for reduction in ('joint_tokens','modalities'):
            kwargs = {'reduction':reduction, 'action_mask':torch.tensor([[1.,0.,1.],[1.,1.,1.]]), 'channel_weights':[1.,2.,3.]}
            a = packing.flow_loss(p,t,'action',50.,1.,**kwargs)
            b = tm.flow_loss(p,t,'action',50.,1.,action_time_mask=torch.ones(2,4,dtype=torch.bool),video_token_mask=torch.ones(2,6,dtype=torch.bool),**kwargs)
            for k in a:
                torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

    def test_masked_targets_zero_gradient_fixed_denominator(self):
        p = {'x_video':torch.ones(1,2,3,requires_grad=True), 'x_action':torch.ones(1,4,3,requires_grad=True)}
        t = {k:torch.zeros_like(v) for k,v in p.items()}
        result = tm.flow_loss(p,t,'action',50.,1.,action_time_mask=torch.tensor([[True,False,False,False]]),video_token_mask=torch.zeros(1,2,dtype=torch.bool))
        torch.testing.assert_close(result['loss'],torch.tensor(50./6))
        result['loss'].backward()
        self.assertEqual(p['x_video'].grad.count_nonzero().item(),0)
        self.assertEqual(p['x_action'].grad[:,1:].count_nonzero().item(),0)
        self.assertGreater(p['x_action'].grad[:,0].abs().sum().item(),0)

    def test_causal_future_not_history_padding_controls_latents(self):
        cfg = SimpleNamespace(inference_profile='history',chunk_size=32,n_obs_steps=8,history_snapshots=2)
        v = torch.zeros(2,40,dtype=torch.bool)
        v[0,8:13] = True  # future0..4: first T32 latent ends at3, second at7
        v[1,8:] = True
        v[1,10] = False  # a hole invalidates all following receptive fields
        a = torch.ones(2,32,dtype=torch.bool)
        am, vm = tm.prepare_masks({'action_time_mask':a,'video_valid_frames':v},[1,0],cfg,torch.zeros(2,96,10,2,3))
        self.assertEqual(vm.sum(1).tolist(),[0,6])
        self.assertEqual(am.shape,(2,32))

    def test_missing_mask_rejected(self):
        cfg = SimpleNamespace(inference_profile='history')
        with self.assertRaises(ValueError):
            tm.prepare_masks({'action_time_mask':torch.ones(1,4,dtype=torch.bool)},[0],cfg,torch.zeros(1))

    def test_endpoints_match_real_temporal_merging_gradients(self):
        from flux_action.models.video_vae import TemporalMerging
        for length, expected_first in [(1,0),(32,3),(33,0),(42,1)]:
            ends=tm.latent_end_indices(length)
            self.assertEqual(ends[0],expected_first)
            x=torch.randn(1,length,1,1,2,requires_grad=True)
            model=torch.nn.Sequential(TemporalMerging(2,2),TemporalMerging(2,2))
            y=model(x)
            self.assertEqual(y.shape[1],len(ends))
            for j,end in enumerate(ends):
                grad=torch.autograd.grad(y[:,j].sum(),x,retain_graph=True)[0]
                active=grad.abs().sum((0,2,3,4)).nonzero().flatten()
                self.assertEqual(int(active.max()),end)
                self.assertEqual(grad[:,end+1:].count_nonzero().item(),0)


if __name__ == '__main__':
    unittest.main()
