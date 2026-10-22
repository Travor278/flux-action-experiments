"""Decode every real frame in the selected 200 episode camera spans on CPU."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
import av

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
meta_path = R/'status/official-v3-inspection.json'
meta = json.loads(meta_path.read_text())
metadata_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
receipts = {v['path']:v for v in map(json.loads,(R/'status/selected-download-receipts.jsonl').read_text().splitlines())}
jobs, files = [], {}
for task, selected in meta['selected'].items():
    for ep in selected['rows']:
        for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
            k = 'videos/observation.images.'+camera
            rel = f"{meta['prefix']}/{k}/chunk-{ep[k+'/chunk_index']:03d}/file-{ep[k+'/file_index']:03d}.mp4"
            if rel not in receipts:
                raise RuntimeError('Wait for verified video: '+rel)
            start = round(ep[k+'/from_timestamp']*25)
            count = round((ep[k+'/to_timestamp']-ep[k+'/from_timestamp'])*25)
            assert 0 <= ep['length']-count <= 1
            job = {'task':task,'episode':ep['episode_index'],'camera':camera,'source':rel,'source_sha256':receipts[rel]['sha256'],'metadata_sha256':metadata_sha,'first_frame':start,'count':count,'numeric_length':ep['length']}
            jobs.append(job)
            files[rel] = receipts[rel]
for rel, rec in files.items():
    path = R/'official-source'/rel
    assert path.stat().st_size == rec['bytes']
    with path.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest() == rec['sha256']

ledger = R/'status/all-video-decode-receipts.jsonl'
key = lambda j: (j['task'],j['episode'],j['camera'])
done = {}
if ledger.exists():
    for line in ledger.read_text().splitlines():
        j = json.loads(line)
        if j['metadata_sha256'] == metadata_sha and files[j['source']]['sha256'] == j['source_sha256']:
            done[key(j)] = j
began = time.time()
def save(complete=False):
    value = {'time':time.time(),'started':began,'complete':complete,'verified_spans':len(done),'total_spans':len(jobs),'decoded_real_frames':sum(v['count'] for v in done.values()),'expected_real_frames':sum(v['count'] for v in jobs),'metadata_sha256':metadata_sha,'source_files':len(files),'scope':'Full CPU decode/PTS/shape across selected real camera frames, RGB samples per span; not semantic cue annotation or expert replay'}
    p = R/'status/all-video-decode-progress.json'
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(value)); tmp.replace(p)
    if complete:
        (R/'status/all-selected-videos-decoded.json').write_text(json.dumps(value,indent=2))

def decode(j):
    path = R/'official-source'/j['source']
    count = 0; samples = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; stream.thread_count = 1
        assert float(stream.average_rate) == 25
        container.seek(int((j['first_frame']/25)/stream.time_base),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            pts = float(frame.pts*stream.time_base)
            if pts < j['first_frame']/25 - .001:
                continue
            assert abs(pts - (j['first_frame']+count)/25) < .001, (j,pts,count)
            assert (frame.width,frame.height) == (640,480)
            if count in (0,j['count']//2,j['count']-1):
                rgb = frame.to_ndarray(format='rgb24')
                assert rgb.shape == (480,640,3) and rgb.std() > 1
                samples.append({'index':count,'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
            count += 1
            if count == j['count']:
                break
    assert count == j['count'], (j,count)
    return {**j,'decoded':count,'rgb_samples':samples,'time':time.time()}

save()
with ledger.open('a') as log, ThreadPoolExecutor(max_workers=2) as pool:
    pending = [pool.submit(decode,j) for j in jobs if key(j) not in done]
    for future in as_completed(pending):
        result = future.result()
        log.write(json.dumps(result)+'\n'); log.flush()
        done[key(result)] = result
        save()
assert len(done) == len(jobs) == 600
save(complete=True)
print('ALL_SELECTED_REAL_VIDEO_FRAMES_DECODED',flush=True)
