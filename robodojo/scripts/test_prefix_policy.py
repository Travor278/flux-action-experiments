import sys
from pathlib import Path
import torch
import pytest

R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
sys.path.insert(0,str(R/'flux-action/tests'))
from conftest import make_policy, FakeVideoVAE
from test_history_training import history_config
from flux_action.policy import FluxActionPolicy
from flux_action.config import PolicyConfig
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.processing import history, prefix_memory as pm


def config(**kw):
    return history_config(fps=25.,chunk_size=32,n_action_steps=2,prefix_memory_spec=pm.PrefixMemorySpec().to_dict(),**kw)


def batch():
    g=torch.Generator().manual_seed(77)
    return {'images.top':torch.randint(256,(1,40,3,64,96),dtype=torch.uint8,generator=g),
            'state':torch.randn(1,8,6,generator=g),'command_history':torch.randn(1,8,6,generator=g),
            'action':torch.randn(1,32,6,generator=g),'task':['remember object'],
            'prefix.images.top':torch.randint(256,(1,3,3,64,96),dtype=torch.uint8,generator=g),
            'prefix_tick_indices':torch.tensor([[0,25,50]]),'episode_tick':torch.tensor([300]),
            'action_time_mask':torch.ones(1,32,dtype=torch.bool),'video_valid_frames':torch.ones(1,40,dtype=torch.bool)}


def test_training_inference_conditioning_exact_and_backward():
    policy=make_policy(config()).eval();data=batch();prepared=policy.prepare(data)
    train={**pm.merge_prefix(history.pack_video(prepared.latents,policy.config),prepared.prefix_tokens,prepared.prefix_ids),**history.pack_actions(prepared.state,prepared.actions,policy.config)}
    infer=pm.inference_conditioning(policy.video_vae,policy._cameras(data)[0,:,:8],data['state'],data['command_history'],policy.config,data,0)
    for k,v in infer.items(): torch.testing.assert_close(train[k],v,rtol=0,atol=0)
    assert prepared.prefix_tokens.shape[1]==18
    assert train['x_video_cond'].shape[1]==30
    assert prepared.prefix_ids[...,0].max()<0
    loss,_=policy(data,prepared=prepared);loss.backward()
    assert torch.isfinite(loss)
    assert policy.dit.final_layer['action'].linear.weight.grad.abs().sum()>0


def test_prefix_can_change_predictions_and_export_reload(tmp_path):
    torch.manual_seed(8);policy=make_policy(config()).eval()
    # Fresh heads normally output zero. Nonzero toy weights test information
    # flow only; this is not a trained robot policy or a real VAE test.
    with torch.no_grad():
        for p in policy.dit.parameters(): p.normal_(0,.1)
    data=batch();data['images.top']=data['images.top'][:,:8]
    a=policy.predict_normalized_targets(data)
    changed={**data,'prefix.images.top':255-data['prefix.images.top']}
    b=policy.predict_normalized_targets(changed)
    assert (a-b).abs().max()>1e-8
    policy.save_pretrained(tmp_path/'export')
    loaded=FluxActionPolicy.from_pretrained(tmp_path/'export',video_vae=FakeVideoVAE(),text_encoder=MockTextEncoder(32))
    assert loaded.config.prefix_memory_spec==policy.config.prefix_memory_spec
    torch.testing.assert_close(a,loaded.predict_normalized_targets(data),rtol=0,atol=0)


def test_queue_ticks_and_reset_reach_prefix_bank(monkeypatch):
    policy=make_policy(config()).eval();calls=[]
    def predict(data):
        calls.append((int(data['episode_tick'][0]),data['prefix.images.top'][0,:,0,0,0].tolist()))
        return torch.zeros(1,32,6)
    monkeypatch.setattr(policy,'predict_normalized_targets',predict)
    for tick in range(28):
        policy.select_action({'images.top':torch.full((1,3,64,96),tick,dtype=torch.uint8),'state':torch.zeros(1,6),'task':['remember']})
    assert calls[-1]==(26,[0,25,26])
    policy.reset()
    policy.select_action({'images.top':torch.full((1,3,64,96),201,dtype=torch.uint8),'state':torch.zeros(1,6),'task':['new episode']})
    assert calls[-1]==(0,[201,201,201])


def test_missing_or_future_prefix_and_augmented_config_fail_closed():
    policy=make_policy(config()).eval();data=batch()
    data['prefix_tick_indices']=torch.tensor([[0,25,301]])
    with pytest.raises(ValueError): policy.prepare(data)
    with pytest.raises(ValueError): config(augment=True)
    bad=config().to_dict();bad['prefix_memory_spec']['version']='unrecognized'
    with pytest.raises(ValueError): PolicyConfig(**bad)
    del data['prefix.images.top'];data['prefix_tick_indices']=torch.tensor([[0,25,50]])
    with pytest.raises(KeyError): policy.prepare(data)
