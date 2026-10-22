"""Read only verified source videos; check selected-episode PTS mapping on CPU."""
from pathlib import Path
import hashlib
import json
import time
import av
import numpy as np

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
meta = json.loads((R / 'status/official-v3-inspection.json').read_text())
receipts = {}
for line in (R / 'status/selected-download-receipts.jsonl').read_text().splitlines():
    item = json.loads(line)
    receipts[item['path']] = item
cameras = ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']
out = {'time': time.time(), 'revision': meta['revision'], 'scope': 'CPU RGB decoding and metadata PTS alignment; not visual cue annotation, action replay, or full video audit', 'tasks': {}, 'source_sha256': {}}
for task, ids in [('cover_blocks', list(range(300, 400))), ('match_and_pick_from_conveyor', list(range(1600, 1700)))]:
    rows = meta['selected'][task]['rows']
    assert sorted(e['episode_index'] for e in rows) == ids
    result = {'episode_count': len(rows), 'available_three_camera_episodes': 0, 'missing_files': [], 'samples': [], 'span_anomalies': []}
    for ep in rows:
        available = True
        for camera in cameras:
            k = 'videos/' + camera
            rel = f"{meta['prefix']}/videos/{camera}/chunk-{ep[k + '/chunk_index']:03d}/file-{ep[k + '/file_index']:03d}.mp4"
            delta = (ep[k + '/to_timestamp'] - ep[k + '/from_timestamp']) * 25 - ep['length']
            if abs(delta) >= 0.01:
                next_ep = next((e for e in rows if e['episode_index'] == ep['episode_index'] + 1), None)
                collision = next_ep is not None and next_ep[k + '/file_index'] == ep[k + '/file_index'] and next_ep[k + '/chunk_index'] == ep[k + '/chunk_index'] and abs(next_ep[k + '/from_timestamp'] - (ep[k + '/from_timestamp'] + (ep['length'] - 1)/25)) < 0.001
                result['span_anomalies'].append({'episode': ep['episode_index'], 'camera': camera, 'length': ep['length'], 'delta_frames': delta, 'from': ep[k + '/from_timestamp'], 'to': ep[k + '/to_timestamp'], 'source': rel, 'last_action_timestamp_equals_next_episode_start': collision})
            if rel not in receipts:
                available = False
                result['missing_files'].append(rel)
        result['available_three_camera_episodes'] += int(available)
    result['missing_files'] = sorted(set(result['missing_files']))
    # Deliberately include episode boundaries rather than merely arbitrary file frames.
    chosen = {rows[0]['episode_index'], rows[len(rows)//2]['episode_index'], rows[-1]['episode_index']}
    chosen.update(x['episode'] for x in result['span_anomalies'])
    for ep in [e for e in rows if e['episode_index'] in chosen]:
        for camera in cameras:
            k = 'videos/' + camera
            rel = f"{meta['prefix']}/videos/{camera}/chunk-{ep[k + '/chunk_index']:03d}/file-{ep[k + '/file_index']:03d}.mp4"
            if rel not in receipts:
                continue
            path = R / 'official-source' / rel
            rec = receipts[rel]
            assert path.stat().st_size == rec['bytes'] and path.stat().st_mtime_ns == rec['mtime_ns']
            if rel not in out['source_sha256']:
                with path.open('rb') as f:
                    digest = hashlib.file_digest(f, 'sha256').hexdigest()
                assert digest == rec['sha256']
                out['source_sha256'][rel] = digest
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                assert abs(float(stream.average_rate) - 25) < 1e-6
                for index in [0, ep['length']//2, ep['length']-1]:
                    target = ep[k + '/from_timestamp'] + index / 25
                    container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
                    matched = None
                    for frame in container.decode(stream):
                        timestamp = float(frame.pts * stream.time_base)
                        if timestamp >= target - 0.001:
                            matched = frame
                            break
                    if matched is None:
                        result['samples'].append({'episode': ep['episode_index'], 'camera': camera, 'frame_index': index, 'target_timestamp': target, 'error': 'no_decoded_frame_at_target'})
                        continue
                    assert abs(timestamp - target) < 0.001, (rel, target, timestamp)
                    rgb = matched.to_ndarray(format='rgb24')
                    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8 and float(rgb.std()) > 1
                    result['samples'].append({'episode': ep['episode_index'], 'camera': camera, 'frame_index': index, 'target_timestamp': target, 'decoded_timestamp': timestamp, 'rgb_std': float(rgb.std()), 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
    base_ids = {rows[0]['episode_index'], rows[len(rows)//2]['episode_index'], rows[-1]['episode_index']}
    result['representative_samples_complete'] = sum(s['episode'] in base_ids and 'error' not in s for s in result['samples']) == 27
    out['tasks'][task] = result
out['sample_decode_passed'] = all(len(t['samples']) > 0 and not any('error' in s for s in t['samples']) for t in out['tasks'].values())
out['metadata_span_alignment_passed'] = all(not t['span_anomalies'] for t in out['tasks'].values())
out['all_selected_videos_available'] = all(t['available_three_camera_episodes'] == 100 for t in out['tasks'].values())
out['representative_samples_complete'] = all(t['representative_samples_complete'] for t in out['tasks'].values())
out['finished'] = time.time()
(R / 'status/video-sample-audit.json').write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k not in ('tasks', 'source_sha256')}), flush=True)
