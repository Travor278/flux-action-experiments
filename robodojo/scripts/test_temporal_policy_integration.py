"""Tiny real policy forward/backward, not a substitute for full-model GPU gates."""
import sys
from pathlib import Path
import torch

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
sys.path.insert(0, str(R/'flux-action/tests'))
from conftest import make_policy
from test_history_training import history_config, batch
from flux_action.processing import temporal_masking


def test_real_prepare_forward_all_valid_matches_unmasked():
    policy = make_policy(history_config()).eval()
    data = batch()
    prepared = policy.prepare(data)
    data['action_time_mask'] = torch.ones(2,42,dtype=torch.bool)
    data['video_valid_frames'] = torch.ones(2,50,dtype=torch.bool)
    masked = policy.prepare(data)
    torch.testing.assert_close(prepared.latents,masked.latents,rtol=0,atol=0)
    torch.manual_seed(100)
    baseline, _ = policy(data, prepared=prepared)
    torch.manual_seed(100)
    result, _ = policy(data, prepared=masked)
    torch.testing.assert_close(baseline,result,rtol=0,atol=0)


def test_prepare_masks_survive_to_real_dit_backward(monkeypatch):
    policy = make_policy(history_config()).train()
    data = batch()
    data['action_time_mask'] = torch.zeros(2,42,dtype=torch.bool)
    data['action_time_mask'][0,:1] = True
    data['action_time_mask'][1,:5] = True
    data['video_valid_frames'] = torch.ones(2,50,dtype=torch.bool)
    data['video_valid_frames'][0,9:] = False
    data['video_valid_frames'][1,13:] = False
    prepared = policy.prepare(data)
    assert prepared.action_time_mask.sum(1).tolist() == [1,5]
    # K42 merges end at1,5,...41; 1 real future frame cannot supervise a latent.
    assert prepared.video_token_mask.sum(1).tolist() == [0,6]
    assert any(t is prepared.action_time_mask for t in prepared.tensors())
    captured = {}
    original = temporal_masking.flow_loss
    def wrapped(pred, *args, **kwargs):
        for key in ('x_video','x_action'):
            pred[key].retain_grad()
            captured[key] = pred[key]
        return original(pred,*args,**kwargs)
    monkeypatch.setattr(temporal_masking,'flow_loss',wrapped)
    loss, _ = policy(data,prepared=prepared)
    loss.backward()
    assert torch.isfinite(loss)
    for key, mask in [('x_action',prepared.action_time_mask),('x_video',prepared.video_token_mask)]:
        grad = captured[key].grad.reshape(mask.shape[0],mask.shape[1],-1)
        assert grad[~mask].count_nonzero().item() == 0
    assert policy.dit.final_layer['action'].linear.weight.grad.abs().sum() > 0
