"""Compact selected-only index for reader/preflight. No training budget freeze."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
meta_path = R/'status/official-v3-inspection.json'
meta = json.loads(meta_path.read_text())
receipts = {x['path']:x for x in map(json.loads,(R/'status/selected-download-receipts.jsonl').read_text().splitlines())}
dest = R/'assets/history-index-preflight-v1'
if dest.exists():
    raise SystemExit('Index exists; inspect before replacing')
tables = {}
sha = lambda p: hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
report = {'time':time.time(),'metadata_sha256':sha(meta_path),'tasks':{},'source_sha256':{},'scope':'Diagnostic chunk32 selected-only index, not frozen training recipe'}
for task, selected in meta['selected'].items():
    episodes, arrays, offset = [], [], 0
    assert len(selected['rows']) == 100
    for ep in selected['rows']:
        rel = f"{meta['prefix']}/data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet"
        path = R/'official-source'/rel
        if rel not in tables:
            digest = sha(path)
            assert digest == receipts[rel]['sha256']
            report['source_sha256'][rel] = digest
            tables[rel] = pq.read_table(path,columns=['episode_index','frame_index','timestamp','observation.state','action'])
        part = tables[rel].filter(pa.compute.equal(tables[rel]['episode_index'],ep['episode_index'])).sort_by('frame_index')
        assert part['frame_index'].to_pylist() == list(range(ep['length']))
        assert np.allclose(part['timestamp'].to_numpy(),np.arange(ep['length'])/25,atol=1e-4,rtol=0)
        state = np.asarray(part['observation.state'].to_pylist(),np.float32)
        action = np.asarray(part['action'].to_pylist(),np.float32)
        assert state.shape == action.shape == (ep['length'],14)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        arrays.append(np.concatenate([state,action],axis=1))
        episodes.append({'episode_index':ep['episode_index'],'episode_id':str(ep['episode_index']),'from_index':offset,'n_frames':ep['length'],'split':'train','caption':ep['tasks'][0],'valid_ranges':[[0,ep['length']]],'source_episode':{k:v for k,v in ep.items() if not k.startswith('stats/')}})
        offset += ep['length']
    out = dest/task
    out.mkdir(parents=True)
    manifest = {'format_version':1,'kind':'lerobot','robodojo_task':task,'source_prefix':meta['prefix'],'dataset_revision':meta['revision'],'fps':25,'chunk_size':32,'action_prev':True,'state_dim':14,'action_dim':14,'total_frames':offset,'camera_order':['head','left_wrist','right_wrist'],'camera_hw':{k:[480,640] for k in ['head','left_wrist','right_wrist']},'episodes':episodes,'warning':'Requires RoboDojoHistoryDataset; default WindowDataset would drop boundaries and lacks per-camera masks'}
    np.save(out/'rows.npy',np.concatenate(arrays))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report['tasks'][task] = {'episodes':len(episodes),'frames':offset,'eligible_anchors':offset,'manifest_sha256':sha(out/'manifest.json'),'rows_sha256':sha(out/'rows.npy')}
report['complete'] = True
(R/'status/history-preflight-index-verified.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
