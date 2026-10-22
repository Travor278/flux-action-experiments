import hashlib,io,json,pathlib,zipfile,time
import h5py,numpy as np
from flux_action.data.lerobot.index import index_dataset, CAPTION_SEPARATOR
from flux_action.training.trainer import TrainConfig
from robotwin_tail_dataset import build
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
report={'time':time.time(),'complete':False,'tasks':{}}
for task in ['click_bell','handover_block']:
    cfg=json.loads((B/f'configs/{task}-10k-opt-v1.json').read_text())
    source=B/f'assets/pi05-v3/{task}';index=B/f'assets/index-tail-v1/{task}'
    assert json.loads((source/'meta/info.json').read_text())['codebase_version']=='v3.0'
    assert not index.exists(), 'Refuse overwriting a prepared repaired index'
    index_dataset(source,index,{'head':'observation.images.cam_head','left_wrist':'observation.images.cam_left_wrist','right_wrist':'observation.images.cam_right_wrist'},
                  action_parameterization='absolute',absolute_action_dims=(),chunk_size=32,val_episodes=0)
    old=np.load(B/f'assets/index/{task}/rows.f32.npy',mmap_mode='r')
    rows=np.load(index/'rows.f32.npy',mmap_mode='r');assert np.array_equal(rows,old),'Data layout changed'
    manifest=json.loads((index/'manifest.json').read_text());assert len(manifest['episodes'])==50
    archive=B/f'assets/robotwin-data/dataset/{task}/demo_clean.zip'
    provenance=[]
    with zipfile.ZipFile(archive) as z:
        names=sorted(x for x in z.namelist() if x.endswith('.hdf5'));assert len(names)==50
        for ep,name in zip(manifest['episodes'],names,strict=True):
            content=z.read(name)
            with h5py.File(io.BytesIO(content)) as h:
                captions=json.loads(h['instructions'][()].decode())
                if isinstance(captions,dict):captions=captions['seen']
                if isinstance(captions,str):captions=[captions]
                assert captions and all(isinstance(x,str) and x.strip() for x in captions)
                # Preserve each source paraphrase once; repeated text must not receive extra weight.
                unique=list(dict.fromkeys(x.strip() for x in captions))
                ep['caption']=CAPTION_SEPARATOR.join(unique)
                provenance.append({'episode':ep['episode_index'],'source':name,'hdf_sha256':hashlib.sha256(content).hexdigest(),'source_captions':len(captions),'unique_captions':len(unique),'captions':unique})
    (index/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (index/'caption-provenance.json').write_text(json.dumps(provenance,indent=2))
    st=json.loads((index/'statistics.json').read_text())
    cfg.update(dataset='robotwin_tail_dataset:build',source_root=str(source),index_dir=str(index),output_dir=str(S/f'runs/{task}-10k-tail-v1'))
    assert not pathlib.Path(cfg['output_dir']).exists()
    for key in ['action','state']:
        cfg['policy'][key+'_normalization']={q:st[key][q] for q in ['q01','q99']}
        previous=json.loads((B/f'assets/index/{task}/statistics.json').read_text())
        for q in ['q01','q99']:assert np.allclose(st[key][q],previous[key][q],rtol=0,atol=1e-7)
    cp=B/f'configs/{task}-10k-tail-v1.json';cp.write_text(json.dumps(cfg,indent=2))
    c=TrainConfig.from_file(cp);assert c.global_batch(2)==32
    ds=build(c,seed=42,epoch=0,rank=0,world_size=2,num_workers=2,windows_per_rank=4,skip_batches=0,grad_accumulation=4)
    real_starts=sum(len(ds.valid_starts(i)) for i in range(50));assert real_starts==len(rows)
    # Real decoder checks first, middle and last observation on two different episodes.
    probes=[]
    for e in [0,49]:
        for start in [0,manifest['episodes'][e]['n_frames']//2,manifest['episodes'][e]['n_frames']-1]:
            w=ds.window_at(e,start);n=manifest['episodes'][e]['n_frames'];off=manifest['episodes'][e]['from_index']
            assert np.array_equal(w['state'].numpy(),rows[off+start,:14])
            assert w['action_time_mask'].sum().item()==min(32,n-start)
            assert w['video_valid_frames']==min(33,n-start)
            if start==n-1:
                assert np.all(w['action'].numpy()==rows[off+start,14:])
                for cam in ds.cameras:assert (w['images.'+cam]==w['images.'+cam][-1:]).all()
            probes.append({'episode':e,'start':start,'valid_actions':int(w['action_time_mask'].sum()),'video_valid_frames':w['video_valid_frames']})
    report['tasks'][task]={'episodes':50,'frames':len(rows),'legal_observation_starts':real_starts,'all_real_rows_unchanged':True,'source_captions':sum(x['source_captions'] for x in provenance),'unique_caption_occurrences':sum(x['unique_captions'] for x in provenance),'decoded_probes':probes,'manifest_sha256':hashlib.sha256((index/'manifest.json').read_bytes()).hexdigest(),'config':str(cp)}
report['complete']=True
(B/'status/tail-repair-data-verified.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
