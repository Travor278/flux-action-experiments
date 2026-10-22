"""Real source CPU acceptance for the reader, NOT acceptance of training loss."""
from pathlib import Path
import hashlib
import json
import time
import av
import numpy as np
import pyarrow.parquet as pq
from robodojo_history_reader import RoboDojoHistoryReader, window_plan

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
meta = json.loads((R/'status/official-v3-inspection.json').read_text())
receipts = {r['path']: r for r in map(json.loads, (R/'status/selected-download-receipts.jsonl').read_text().splitlines())}
out = {'time': time.time(), 'scope': 'CPU reader only; H=8/64 diagnostic settings, not frozen Memory training configuration; no full-model/masked loss acceptance', 'planned_windows': 0, 'decoded_windows': [], 'source_sha256': {}}

def check_source(path):
    rel = str(path.relative_to(R/'official-source'))
    rec = receipts[rel]
    if rel not in out['source_sha256']:
        assert path.stat().st_size == rec['bytes']
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        assert digest == rec['sha256']
        out['source_sha256'][rel] = digest

def decoder(path, timestamps):
    check_source(path)
    found = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for target in sorted(set(timestamps)):
            container.seek(int(target/stream.time_base), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                pts = float(frame.pts*stream.time_base)
                if pts >= target-0.001:
                    assert abs(pts-target) < 0.001
                    found[target] = frame.to_ndarray(format='rgb24').transpose(2,0,1)
                    break
            assert target in found
    return np.stack([found[t] for t in timestamps])

for task, selected in meta['selected'].items():
    for ep in selected['rows']:
        for h in (8, 64):
            for start in (0, ep['length']//2, ep['length']-1):
                p = window_plan(ep, start, h, 32)
                assert all(i <= start for i in p['state_indices'])
                assert all(i is None or i < start for i in p['command_indices'])
                assert sum(p['action_time_mask']) == min(32, ep['length']-start)
                out['planned_windows'] += 1
    # Actual missing-camera endpoint case plus a normal reset for each task,
    # whenever that episode's entire three-camera source is SHA verified.
    targets = [selected['rows'][0], next(e for e in selected['rows'] if e['episode_index'] == (324 if task == 'cover_blocks' else 1625))]
    for ep in targets:
        for cam in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            k = 'videos/observation.images.'+cam
            rel = f"{meta['prefix']}/videos/observation.images.{cam}/chunk-{ep[k+'/chunk_index']:03d}/file-{ep[k+'/file_index']:03d}.mp4"
            if rel not in receipts:
                break
        else:
            rel = f"{meta['prefix']}/data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet"
            path = R/'official-source'/rel
            check_source(path)
            table = pq.read_table(path, filters=[('episode_index', '=', ep['episode_index'])], columns=['frame_index','observation.state','action'])
            table = table.sort_by('frame_index')
            assert table['frame_index'].to_pylist() == list(range(ep['length']))
            states = np.array(table['observation.state'].to_pylist(), dtype=np.float32)
            actions = np.array(table['action'].to_pylist(), dtype=np.float32)
            reader = RoboDojoHistoryReader(ep, task, states, actions, R/'official-source', meta['prefix'], decoder)
            for start in (0, ep['length']-1):
                item = reader.window_at(start, history=8, chunk=32)
                np.testing.assert_array_equal(item['action'][0], actions[start])
                np.testing.assert_array_equal(item['state'][-1], states[start])
                assert item['action_time_mask'].sum() == min(32,ep['length']-start)
                record = {'task': task, 'episode': ep['episode_index'], 'start': start, 'valid_actions': int(item['action_time_mask'].sum()), 'valid_video_frames': int(item['video_valid_frames'].sum()), 'rgb': {key: hashlib.sha256(item[key].tobytes()).hexdigest() for key in ('images.head','images.left_wrist','images.right_wrist')}}
                out['decoded_windows'].append(record)
                del item
out['reader_cpu_verified'] = out['planned_windows'] == 1200 and len(out['decoded_windows']) >= 4
out['both_tasks_decoded'] = len({v['task'] for v in out['decoded_windows']}) == 2
out['trainer_loss_integration_verified'] = False
out['finished'] = time.time()
(R/'status/history-reader-cpu-audit.json').write_text(json.dumps(out, indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ('decoded_windows','source_sha256')}), flush=True)
