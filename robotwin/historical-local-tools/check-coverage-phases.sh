set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,numpy as np,time
from flux_action.training.data import WindowDataset
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');out={'time':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 I=B/'assets/index'/task;m=json.loads((I/'manifest.json').read_text());D=WindowDataset(m,str(B/'assets/lerobot'/task),I/'rows.f32.npy',seed=42,decoder='pyav',frame_hw=[240,320]);data=np.load(I/'rows.f32.npy',mmap_mode='r');rows=[]
 for i,e in enumerate(m['episodes']):
  x=data[e['from_index']:e['from_index']+e['n_frames'],:14];starts=D.valid_starts(i);closed=x[:,[6,13]]<.2;cl=closed[:,0].astype(int)+2*closed[:,1].astype(int)
  right_was_closed=np.maximum.accumulate(closed[:,1]);right_opening=right_was_closed & (np.r_[0,np.diff(x[:,13])]>1e-5)
  closing_seen=np.flatnonzero(closed[:,1]);after=right_was_closed & (x[:,13]>.8)
  rows.append({'episode':i,'frames':len(x),'valid_starts':len(starts),'last_valid':max(starts),
    'all_categories':[int((cl==k).sum()) for k in range(4)],'sample_categories':[float((cl[starts]==k).mean()) for k in range(4)],
    'right_opening_after_grasp_frames':int(right_opening.sum()),'right_opening_after_grasp_valid_starts':int(right_opening[starts].sum()),
    'first_right_release_frame':int(np.flatnonzero(right_opening)[0]) if right_opening.any() else None,
    'fully_right_open_after_grasp_frames':int(after.sum()),'fully_right_open_after_grasp_valid_starts':int(after[starts].sum())})
 out['tasks'][task]={'categories':['both_open','left_closed_only','right_closed_only','both_closed'],'sampler_probabilities':np.mean([r['sample_categories'] for r in rows],axis=0).tolist(),
  'right_release_frames':sum(r['right_opening_after_grasp_frames'] for r in rows),'right_release_valid_starts':sum(r['right_opening_after_grasp_valid_starts'] for r in rows),
  'right_open_after_grasp_frames':sum(r['fully_right_open_after_grasp_frames'] for r in rows),'right_open_after_grasp_valid_starts':sum(r['fully_right_open_after_grasp_valid_starts'] for r in rows),'per_episode':rows}
(B/'status/coverage-phase-detail.json').write_text(json.dumps(out,indent=2))
PY
