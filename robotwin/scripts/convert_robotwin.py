"""Lossless RGB and explicit state/action conversion of pinned RoboTwin archives."""
import io
import json
import pathlib
import sys
import zipfile
import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

B = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))
from XPolicyLab.utils.process_data import decode_image_bit

CAMERAS = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']
JOINTS = ['left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states']

def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))

def video_write(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('libx264rgb', rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = 'rgb24'
        stream.options = {'crf': '0', 'preset': 'fast', 'g': '16', 'threads': '2'}
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    with av.open(str(path)) as src:
        decoded = np.stack([f.to_ndarray(format='rgb24') for f in src.decode(video=0)])
    if not np.array_equal(decoded, frames):
        raise ValueError(f'RGB roundtrip mismatch {path}')

def convert(task):
    source = B/'assets/robotwin-data/dataset'/task/'demo_clean.zip'
    final = B/'assets/lerobot'/task
    if (final/'CONVERSION_COMPLETE.json').exists():
        return
    out = final.with_name('.' + task + '.converting')
    (out/'meta').mkdir(parents=True, exist_ok=True)
    (out/'data/chunk-000').mkdir(parents=True, exist_ok=True)
    tasks, episodes, arrays, audits = {}, [], {'state': [], 'action': []}, []
    offset = 0
    with zipfile.ZipFile(source) as archive:
        files = sorted(n for n in archive.namelist() if n.endswith('.hdf5'))
        assert len(files) == 50, len(files)
        for ei, name in enumerate(files):
            with h5py.File(io.BytesIO(archive.read(name))) as h:
                state = np.concatenate([h[f'state/{k}'][()] for k in JOINTS], axis=1).astype('float32')
                action = np.concatenate([h[f'action/{k}'][()] for k in JOINTS], axis=1).astype('float32')
                n = len(state)
                assert state.shape == action.shape == (n, 14)
                assert np.isfinite(state).all() and np.isfinite(action).all()
                fps = int(h['additional_info/frequency'][()])
                assert fps == 15, fps
                captions = json.loads(h['instructions'][()].decode())
                if isinstance(captions, dict):
                    captions = captions['seen']
                if isinstance(captions, str):
                    captions = [captions]
                assert captions and all(isinstance(x, str) and x for x in captions)
                caption = captions[ei % len(captions)]
                ti = tasks.setdefault(caption, len(tasks))
                shape = None
                for camera in CAMERAS:
                    frames = decode_image_bit(h[f'vision/{camera}/colors'][()])
                    assert len(frames) == n and frames.dtype == np.uint8
                    shape = frames.shape[1:]
                    video_write(out/f'videos/chunk-000/observation.images.{camera}/episode_{ei:06d}.mp4', frames, fps)
                table = pa.table({'observation.state': state.tolist(), 'action': action.tolist(),
                                  'timestamp': np.arange(n, dtype='float32')/fps,
                                  'frame_index': np.arange(n, dtype='int64'),
                                  'episode_index': np.full(n, ei, dtype='int64'),
                                  'index': np.arange(offset, offset+n, dtype='int64'),
                                  'task_index': np.full(n, ti, dtype='int64')})
                pq.write_table(table, out/f'data/chunk-000/episode_{ei:06d}.parquet')
                episodes.append({'episode_index': ei, 'tasks': [caption], 'length': n})
                arrays['state'].append(state); arrays['action'].append(action)
                audits.append({'episode': ei, 'source': name, 'frames': n,
                               'left_joint_range': float(np.ptp(state[:,:6], axis=0).max()),
                               'right_joint_range': float(np.ptp(state[:,7:13], axis=0).max())})
                offset += n
                print(task, ei, n, 'RGB verified', flush=True)
    features = {key: {'dtype': 'float32', 'shape': [14], 'names': [f'joint_{i}' for i in range(14)]}
                for key in ['observation.state', 'action']}
    for key in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
        features[key] = {'dtype': 'float32' if key == 'timestamp' else 'int64', 'shape': [1], 'names': None}
    features.update({f'observation.images.{c}': {'dtype': 'video', 'shape': list(shape), 'names': ['height','width','channels'],
                      'info': {'video.fps': fps, 'video.codec': 'h264', 'video.pix_fmt': 'gbrp'}} for c in CAMERAS})
    info = {'codebase_version': 'v2.1', 'robot_type': 'aloha_agilex', 'fps': fps,
            'total_episodes': len(files), 'total_frames': offset, 'total_tasks': len(tasks), 'total_videos': len(files)*3,
            'total_chunks': 1, 'chunks_size': 1000, 'splits': {'train': f'0:{len(files)}'}, 'features': features,
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}
    (out/'meta/info.json').write_text(json.dumps(info, indent=2))
    write_jsonl(out/'meta/episodes.jsonl', episodes)
    write_jsonl(out/'meta/tasks.jsonl', [{'task_index': i, 'task': t} for t, i in tasks.items()])
    stats={}
    for key, rows in arrays.items():
        x=np.concatenate(rows)
        stats['observation.state' if key=='state' else key] = {k:v.tolist() for k,v in {
            'mean':x.mean(0),'std':x.std(0),'min':x.min(0),'max':x.max(0),'q01':np.quantile(x,.01,axis=0),'q99':np.quantile(x,.99,axis=0)}.items()}
    (out/'meta/stats.json').write_text(json.dumps(stats))
    (out/'CONVERSION_COMPLETE.json').write_text(json.dumps({'task':task,'episodes':audits,'rgb_roundtrip':'all frames exact',
        'action_semantics':'native explicit action; no temporal shift or gripper flip',
        'source_revision':json.loads((B/'assets/robotwin-data.manifest.json').read_text())['revision']}, indent=2))
    out.rename(final)

if __name__ == '__main__':
    for task in sys.argv[1:] or ['click_bell','handover_block']:
        convert(task)
