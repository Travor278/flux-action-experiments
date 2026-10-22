"""Prepare separate v2.1 inputs for the official LeRobot v3 converter.

Never edits the FLUX dataset. Numeric/video files are immutable hardlinks;
metadata is independently copied and genuine per-episode statistics are added.
"""
import json,os,pathlib,shutil
import av
import numpy as np
import pyarrow.parquet as pq

B=pathlib.Path(__file__).resolve().parents[1]

def vector_stats(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==1:x=x[:,None]
    stats={'min':x.min(0),'max':x.max(0),'mean':x.mean(0),'std':x.std(0),'count':np.array([len(x)])}
    stats.update({name:np.quantile(x,q,axis=0) for name,q in [('q01',.01),('q10',.1),('q50',.5),('q90',.9),('q99',.99)]})
    return {k:v.tolist() for k,v in stats.items()}

def image_stats(path):
    hist=np.zeros((3,256),dtype=np.int64);n=0;pixels=0
    with av.open(str(path)) as source:
        for frame in source.decode(video=0):
            rgb=frame.to_ndarray(format='rgb24');n+=1;pixels+=rgb.shape[0]*rgb.shape[1]
            for c in range(3):hist[c]+=np.bincount(rgb[:,:,c].ravel(),minlength=256)
    assert n>0
    levels=np.arange(256)/255
    mean=(hist*levels).sum(1)/pixels
    stats={'mean':mean,'std':np.sqrt(np.maximum(0,(hist*levels**2).sum(1)/pixels-mean**2)),
           'min':np.array([np.flatnonzero(h)[0]/255 for h in hist]),
           'max':np.array([np.flatnonzero(h)[-1]/255 for h in hist])}
    cdf=hist.cumsum(1)
    stats.update({name:np.array([np.searchsorted(h,q*pixels)/255 for h in cdf]) for name,q in [('q01',.01),('q10',.1),('q50',.5),('q90',.9),('q99',.99)]})
    return {**{k:v.reshape(3,1,1).tolist() for k,v in stats.items()},'count':[n]},n

def main():
    receipts={}
    for task in ['click_bell','handover_block']:
        source=B/'assets/lerobot'/task
        root=B/'assets/pi05-v21'/task
        marker=root/'PI05_V21_PREPARED.json'
        if marker.exists():continue
        if root.exists():raise FileExistsError('Incomplete prior copy; inspect before retry: '+str(root))
        root.parent.mkdir(parents=True,exist_ok=True)
        def copy_file(src,dst):
            if 'meta' in pathlib.Path(src).relative_to(source).parts:return shutil.copy2(src,dst)
            os.link(src,dst);return dst
        shutil.copytree(source,root,copy_function=copy_file)
        info=json.loads((root/'meta/info.json').read_text())
        videos=[k for k,v in info['features'].items() if v['dtype']=='video']
        # LeRobot's policy feature declarations use CHW while stored MP4 remains RGB.
        for key in videos:
            feature=info['features'][key]
            shape=dict(zip(feature['names'],feature['shape']))
            feature['shape']=[shape['channels'],shape['height'],shape['width']]
            feature['names']=['channels','height','width']
        (root/'meta/info.json').write_text(json.dumps(info,indent=2))
        episodes=[json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines()]
        with (root/'meta/episodes_stats.jsonl').open('w') as output:
            for ep in episodes:
                ei=ep['episode_index'];n=ep['length']
                p=root/info['data_path'].format(episode_chunk=ei//1000,episode_index=ei)
                columns=pq.read_table(p).to_pydict()
                stats={k:vector_stats(v) for k,v in columns.items()}
                for key in videos:
                    p=root/info['video_path'].format(episode_chunk=ei//1000,episode_index=ei,video_key=key)
                    stats[key],decoded=image_stats(p)
                    if decoded!=n:raise ValueError((task,ei,key,decoded,n))
                output.write(json.dumps({'episode_index':ei,'stats':stats})+'\n')
                output.flush();print(task,ei,'real per-episode stats complete',flush=True)
        record={'source':str(source),'episodes':len(episodes),'frames':info['total_frames'],
                'stage':'ready for official v2.1 to v3 conversion, not yet a validated pi05 dataset',
                'numeric_and_video_content':'identical hardlinks; metadata copied independently',
                'video_stats':'all frames, exact per-channel pixel histograms',
                'source_revision':json.loads((source/'CONVERSION_COMPLETE.json').read_text())['source_revision']}
        marker.write_text(json.dumps(record,indent=2));receipts[task]=record
    (B/'status/pi05-v21-prepared.json').write_text(json.dumps(receipts,indent=2))

if __name__=='__main__':main()
