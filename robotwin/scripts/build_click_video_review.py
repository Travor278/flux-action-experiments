import pathlib,json,subprocess,time,html
from PIL import Image,ImageDraw
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/video-review-10k-click';R.mkdir(exist_ok=True)
i=json.loads((B/'status/flux-video-cleanup-inspection.json').read_text());V=pathlib.Path(i['video_dir']);pairs=json.loads((B/'evaluation/click_bell/flux-10k-test100/paired_seeds.json').read_text());cards=[];info=[]
for n in [19,58,0,18,99]:
 v=V/f'episode{n}.mp4';duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(v)],text=True));times=[duration*f for f in [.05,.30,.60,.95]];images=[]
 for k,t in enumerate(times):
  p=R/f'frame-{n}-{k}.jpg';subprocess.run(['ffmpeg','-v','error','-y','-threads','1','-ss',str(t),'-i',str(v),'-frames:v','1','-threads','1',str(p)],check=True);images.append(Image.open(p).convert('RGB'))
 w,h=images[0].size;strip=Image.new('RGB',(w*4,h+26),'#121827');draw=ImageDraw.Draw(strip)
 for k,(im,t) in enumerate(zip(images,times)):strip.paste(im,(w*k,26));draw.text((w*k+8,6),f'{t:.1f}s / {duration:.1f}s',fill='white')
 strip.save(R/f'strip-{n}.jpg',quality=90)
 for k in range(4):(R/f'frame-{n}-{k}.jpg').unlink()
 pair=pairs[n];url='../../'+str(v.relative_to(B));result='SUCCESS' if pair['success'] else 'FAILURE';info.append({'episode':n,'seed':pair['seed'],'success':pair['success'],'duration':duration,'instruction':pair['instruction'],'source':str(v)})
 cards.append(f'<section id="episode{n}"><h2>{result} · episode {n} · seed {pair["seed"]}</h2><p>{html.escape(pair["instruction"])}</p><img src="strip-{n}.jpg" width="1280"><details><summary>播放完整视频 ({duration:.1f}s)</summary><video controls preload="metadata" width="640" src="{html.escape(url)}"></video></details><a href="{html.escape(url)}">原始视频</a></section>')
(R/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>FLUX 10k click_bell 视频复核</title><style>body{margin:24px;background:#f5f6fa;color:#17223b;font-family:system-ui;max-width:1300px}section{background:white;padding:18px;margin:18px 0;border-radius:12px}h2{font-size:18px}p{font-size:14px}img{max-width:100%;height:auto}summary{padding:12px;cursor:pointer}video{max-width:100%}a{color:#1751aa}</style><h1>click_bell · FLUX 10k</h1><p>最终100回合成功6次。下列为2个成功与3个失败样例；逐帧预览不替代完整视频，官方回合判定已核验。</p>'+''.join(cards)+'</html>')
(B/'status/click-video-review-ready.json').write_text(json.dumps({'time':time.time(),'complete':True,'viewer':str(R/'index.html'),'cases':info},indent=2))
