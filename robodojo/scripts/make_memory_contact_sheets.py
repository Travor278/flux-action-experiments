"""Remote-only training-video review sheets; no privileged simulator labels."""
from pathlib import Path
import json
import hashlib
import time
import av
from PIL import Image, ImageDraw

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
meta = json.loads((R/'status/official-v3-inspection.json').read_text())
out = R/'status/memory-contact-sheets'
out.mkdir(exist_ok=True)
manifest = {'time':time.time(),'scope':'Train episodes only, RGB contact sheets for manual cue review; not GT policy input','sheets':[]}
for task, ids in [('cover_blocks',[300,350]),('match_and_pick_from_conveyor',[1600,1650])]:
    for ep in (e for e in meta['selected'][task]['rows'] if e['episode_index'] in ids):
        k='videos/observation.images.cam_high'
        path = R/'official-source'/f"{meta['prefix']}/{k}/chunk-{ep[k+'/chunk_index']:03d}/file-{ep[k+'/file_index']:03d}.mp4"
        count = min(ep['length'],round((ep[k+'/to_timestamp']-ep[k+'/from_timestamp'])*25))
        indices = list(range(0,count,25))
        if indices[-1] != count-1: indices.append(count-1)
        w,h = 320,264
        sheet = Image.new('RGB',(w*4,h*((len(indices)+3)//4)),(22,22,22))
        draw = ImageDraw.Draw(sheet)
        frames=[]
        with av.open(str(path)) as container:
            stream = container.streams.video[0];stream.thread_count=1
            for n,index in enumerate(indices):
                target = ep[k+'/from_timestamp']+index/25
                container.seek(int(target/stream.time_base),stream=stream,backward=True,any_frame=False)
                for frame in container.decode(stream):
                    pts=float(frame.pts*stream.time_base)
                    if pts>=target-.001:
                        assert abs(pts-target)<.001
                        im=frame.to_image();break
                x,y=(n%4)*w,(n//4)*h
                sheet.paste(im.resize((320,240)),(x,y+24))
                draw.text((x+5,y+5),f'{task} ep{ep["episode_index"]} f{index} t={index/25:.2f}s',fill='white')
                frames.append(index)
        name=f'{task}-{ep["episode_index"]}.jpg'
        sheet.save(out/name,quality=90)
        manifest['sheets'].append({'task':task,'episode':ep['episode_index'],'source':str(path),'indices':frames,'sheet':name,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('TRAIN_VIDEO_CONTACT_SHEETS_READY',flush=True)
