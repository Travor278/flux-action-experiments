"""Run the official LeRobot 0.4.3 converter on a separate copy, then audit it."""
import importlib.metadata,json,os,pathlib,shutil
import av
import numpy as np
import pyarrow.parquet as pq

B=pathlib.Path(__file__).resolve().parents[1]

def main():
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    version=importlib.metadata.version('lerobot')
    if version!='0.4.3':raise RuntimeError('Expected exact LeRobot 0.4.3, found '+version)
    results={}
    for task in ['click_bell','handover_block']:
        source=B/'assets/lerobot'/task
        prepared=B/'assets/pi05-v21'/task
        dest=B/'assets/pi05-v3'/task
        marker=dest/'PI05_DATA_VERIFIED.json'
        if marker.exists():
            results[task]=json.loads(marker.read_text());continue
        if not (prepared/'PI05_V21_PREPARED.json').exists():raise RuntimeError('Per-episode statistics not ready')
        if any(p.exists() for p in [dest,dest.with_name(task+'_old'),dest.with_name(task+'_v30')]):
            raise RuntimeError('Prior incomplete conversion exists; inspect before retry')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.resolve().is_relative_to((B/'assets/pi05-v3').resolve()):raise ValueError('Unsafe output path')
        def copy(src,dst):
            if 'meta' in pathlib.Path(src).relative_to(prepared).parts:return shutil.copy2(src,dst)
            os.link(src,dst);return dst
        shutil.copytree(prepared,dest,copy_function=copy)
        convert_dataset(repo_id=task,root=dest.parent,push_to_hub=False)
        # Preserve exact global training quantiles; averaging per-episode quantiles is not a global quantile.
        canonical=json.loads((source/'meta/stats.json').read_text())
        stats=json.loads((dest/'meta/stats.json').read_text())
        for key in ['observation.state','action']:stats[key].update(canonical[key])
        (dest/'meta/stats.json').write_text(json.dumps(stats))
        expected=[pq.read_table(p).to_pydict() for p in sorted((source/'data').rglob('*.parquet'))]
        converted=[pq.read_table(p).to_pydict() for p in sorted((dest/'data').rglob('*.parquet'))]
        for key in ['observation.state','action','timestamp','index','frame_index','episode_index','task_index']:
            a=np.concatenate([np.asarray(x[key]) for x in expected])
            b=np.concatenate([np.asarray(x[key]) for x in converted])
            np.testing.assert_array_equal(a,b,err_msg=task+': '+key)
        ds=LeRobotDataset(task,root=dest,video_backend='pyav',download_videos=False)
        info=json.loads((source/'meta/info.json').read_text())
        episodes=[json.loads(x) for x in (source/'meta/episodes.jsonl').read_text().splitlines()]
        offset=0;checked=0
        for ep in episodes:
            ei=ep['episode_index'];n=ep['length']
            for fi in [0,n-1]:
                row=ds[offset+fi]
                np.testing.assert_array_equal(row['observation.state'].numpy(),np.asarray(expected[ei]['observation.state'][fi],dtype=np.float32))
                np.testing.assert_array_equal(row['action'].numpy(),np.asarray(expected[ei]['action'][fi],dtype=np.float32))
                if row['task']!=ep['tasks'][0]:raise ValueError('Language mapping differs')
                for key in ds.meta.camera_keys:
                    p=source/info['video_path'].format(episode_chunk=ei//1000,episode_index=ei,video_key=key)
                    with av.open(str(p)) as video:
                        for index,frame in enumerate(video.decode(video=0)):
                            if index==fi:rgb=frame.to_ndarray(format='rgb24');break
                    reference=np.ascontiguousarray(rgb.transpose(2,0,1)).astype(np.float32)/255
                    np.testing.assert_allclose(row[key].numpy(),reference,atol=1e-6,rtol=0)
                checked+=1
            offset+=n
        if len(ds)!=info['total_frames'] or ds.num_episodes!=50:raise ValueError('Dataset size differs')
        record={'task':task,'lerobot':version,'root':str(dest),'frames':len(ds),'episodes':ds.num_episodes,
                'all_numeric_columns':'exact match to FLUX source','decoded_boundary_rows':checked,
                'all_three_rgb_views':'source exact within 1e-6 after uint8 to float32',
                'normalization':'original global training state/action quantiles preserved',
                'source_revision':json.loads((source/'CONVERSION_COMPLETE.json').read_text())['source_revision']}
        marker.write_text(json.dumps(record,indent=2));results[task]=record
        print(json.dumps(record),flush=True)
    (B/'status/pi05-data-verified.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
