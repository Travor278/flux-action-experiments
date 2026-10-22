set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/sim-venv/bin/python" - <<'PY'
import pathlib,cv2
from PIL import Image,ImageDraw
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
root=B/'RoboTwin/eval_result/handover_block/FLUX_Action/demo_clean/flux3000-validation20'
videos=sorted(root.rglob('episode0.mp4'))+sorted(root.rglob('episode7.mp4'))+sorted(root.rglob('episode17.mp4'))
assert len(videos)==3
canvas=Image.new('RGB',(6*320,3*270),'white');draw=ImageDraw.Draw(canvas)
for row,p in enumerate(videos):
 cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 for col,f in enumerate([0,.2,.4,.6,.8,1]):
  i=min(n-1,int(f*(n-1)));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,a=cap.read();assert ok
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((320,240));canvas.paste(im,(col*320,row*270+25));draw.text((col*320+3,row*270+5),f'{p.stem} frame {i}/{n}',fill='black')
 cap.release()
canvas.save(B/'status/handover-validation-contact.jpg',quality=85)
PY
