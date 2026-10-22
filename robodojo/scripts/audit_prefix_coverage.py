"""Training-only fixed-anchor coverage; semantic review remains separate."""
from pathlib import Path
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
from flux_action.data.droid.video import decode_window
from prefix_memory import PrefixMemorySpec

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
meta = json.loads((R/'status/official-v3-inspection.json').read_text())
spec = PrefixMemorySpec()
root = R/'status/memory-contact-sheets'
root.mkdir(exist_ok=True)
out = {'time':time.time(),'candidate_spec':spec.to_dict(),'scope':'All200train episode fixedRGB anchors decode/availability; initialheadcam montage25/task for separate manual review, not a semantic visibility oracle','episodes':[]}

def one(pair):
    task,ep = pair
    result = {'task':task,'episode':ep['episode_index'],'cameras':{}}
    thumbnail = None
    for cam in ('cam_high','cam_left_wrist','cam_right_wrist'):
        k='videos/observation.images.'+cam
        count=round((ep[k+'/to_timestamp']-ep[k+'/from_timestamp'])*25)
        assert count > max(spec.ticks)
        path=R/'official-source'/f"{meta['prefix']}/{k}/chunk-{ep[k+'/chunk_index']:03d}/file-{ep[k+'/file_index']:03d}.mp4"
        start=round(ep[k+'/from_timestamp']*25)
        frames=decode_window(path,start,max(spec.ticks)+1,fps=25,frame_hw=(480,640),decoder='pyav',threads=1)
        rgb=frames[list(spec.ticks)]
        assert rgb.dtype==np.uint8 and all(frame.std()>1 for frame in rgb)
        result['cameras'][cam]={'ticks':list(spec.ticks),'rgb_sha256':[hashlib.sha256(frame.tobytes()).hexdigest() for frame in rgb]}
        if cam=='cam_high' and (ep['episode_index']%100)%4==0:
            thumbnail=Image.fromarray(rgb[0]).resize((320,240))
    return result,thumbnail

jobs=[(task,ep) for task,s in meta['selected'].items() for ep in s['rows']]
sheets={task:Image.new('RGB',(1600,1320),(20,20,20)) for task in meta['selected']}
with ThreadPoolExecutor(max_workers=2) as pool:
    for result,thumbnail in pool.map(one,jobs):
        out['episodes'].append(result)
        if thumbnail is not None:
            i=(result['episode']%100)//4;x,y=(i%5)*320,(i//5)*264
            sheet=sheets[result['task']];sheet.paste(thumbnail,(x,y+24))
            ImageDraw.Draw(sheet).text((x+5,y+5),f"{result['task']} episode{result['episode']} tick0",fill='white')
        (R/'status/prefix-coverage-progress.json').write_text(json.dumps({'time':time.time(),'episodes':len(out['episodes']),'total':200}))
out['sheets']={}
for task,sheet in sheets.items():
    path=root/f'{task}-initial-25.jpg';sheet.save(path,quality=92)
    out['sheets'][task]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
out['complete']=len(out['episodes'])==200
out['finished']=time.time()
(R/'status/prefix-anchor-coverage.json').write_text(json.dumps(out,indent=2))
print('PREFIX_200_EPISODES_RGB_COVERAGE_PASS',flush=True)
