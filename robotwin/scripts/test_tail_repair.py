import numpy as np
import torch
from flux_action.processing.packing import flow_loss
from conftest import make_policy,tiny_config
from flux_action.training.data import collate_windows
from robotwin_tail_dataset import TailWindows

def test_temporal_padding_has_no_loss_or_gradient_and_full_masks_match():
    p={'x_video':torch.randn(1,16,3,requires_grad=True),'x_action':torch.randn(1,8,7,requires_grad=True)}
    y={k:torch.randn_like(v) for k,v in p.items()}
    a=torch.tensor([[1,1,1,1],[1,0,0,0]],dtype=torch.bool)
    v=torch.tensor([[1]*8,[0]*8],dtype=torch.bool)
    masked=flow_loss(p,y,'action',action_time_mask=a,video_time_mask=v)
    masked['loss'].backward()
    assert (p['x_action'].grad[:,5:]==0).all() and (p['x_video'].grad[:,8:]==0).all()
    changed={k:x.clone() for k,x in y.items()};changed['x_action'][:,5:]=1e9;changed['x_video'][:,8:]=1e9
    assert torch.equal(masked['loss'],flow_loss(p,changed,'action',action_time_mask=a,video_time_mask=v)['loss'])
    full=flow_loss(p,y,'action',action_time_mask=torch.ones_like(a),video_time_mask=torch.ones_like(v))
    old=flow_loss(p,y,'action');assert torch.equal(full['loss'],old['loss'])

def test_tail_survives_preparation_video_latent_mask_and_backward():
    policy=make_policy(tiny_config(action_dim=14))
    b={'images.top':torch.randint(0,256,(3,33,3,64,96),dtype=torch.uint8),
       'state':torch.rand(3,14),'action':torch.rand(3,32,14),'task':['a','b','c'],
       'action_time_mask':torch.arange(32)[None,:]<torch.tensor([32,5,1])[:,None],
       'video_valid_frames':torch.tensor([33,5,1])}
    prep=policy.prepare(b);assert len(prep.idx)==3
    assert prep.action_time_mask.sum(-1).tolist()==[32,5,1]
    assert prep.video_time_mask.sum(-1).tolist()==[8*6,6,0]
    loss,info=policy(b,prepared=prep);assert info['n_valid_windows']==3 and torch.isfinite(loss)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)

def test_all_tail_rows_never_cross_episode_and_resume_is_deterministic(monkeypatch,tmp_path):
    import robotwin_tail_dataset as module
    def decode(path,first,count,**kwargs):
        return np.full((count,2,2,3),int(path[-1]),np.uint8)
    monkeypatch.setattr(module,'decode_window',decode)
    episodes=[{'episode_index':i,'episode_id':str(i),'n_frames':40,'from_index':i*40,'caption':'instruction',
               'valid_ranges':[[1,40]],'videos':{'top':{'file':'camera'+str(i),'first_frame':0}}} for i in range(4)]
    rows=np.concatenate([np.full((40,28),i,np.float32) for i in range(4)]);np.save(tmp_path/'rows.npy',rows)
    m={'episodes':episodes,'fps':15,'chunk_size':32,'total_frames':160,'state_dim':14,'action_dim':14,'action_prev':True,'camera_order':['top']}
    kw=dict(seed=42,num_workers=1,windows_per_rank=2,grad_accumulation=1,visits_per_epoch=4,frame_hw=(2,2))
    ds=TailWindows(m,tmp_path,tmp_path/'rows.npy',**kw)
    for ep in range(4):
        for start in range(40):
            sample=ds.window_at(ep,start)
            assert (sample['action']==ep).all() and (sample['state']==ep).all()
            assert int(sample['action_time_mask'].sum())==min(32,40-start)
    full=list(ds);resumed=list(TailWindows(m,tmp_path,tmp_path/'rows.npy',skip_batches=3,**kw))
    assert [(x['episode_index'],x['start'],x['task'],x['window_seed']) for x in full[6:]]==[(x['episode_index'],x['start'],x['task'],x['window_seed']) for x in resumed]
