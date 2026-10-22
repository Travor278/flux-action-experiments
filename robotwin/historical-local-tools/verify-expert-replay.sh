set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,hashlib,time,numpy as np,subprocess,zipfile,html
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');R=B/'evaluation/expert-replay-click';p=json.loads((R/'plan.json').read_text());s=json.loads((B/'status/expert-replay-summary.json').read_text());assert s['complete'] and s['completed']==s['requested']==s['recorded_successes']==10
out={'time':time.time(),'complete':True,'diagnostic_only':True,'model_used':False,'source_seeds_verified':True,'results':[],'successes':0,'left':0,'right':0,'camera_note':'Raw stored4x4 extrinsic is not directly numerically comparable to runtime3x4 CV matrix; pixel/scene/command checks are separate from convention metadata'};rgb=[]
for rec,r in zip(p['episodes'],s['results']):
 assert r['episode']==rec['episode'] and r['seed']==rec['seed'] and r['source_hdf5_sha256']==rec['source_hdf5_sha256'] and r['replay_success'] and r['complete'] and not r['model_used'] and r['scene_identity_matches'] and 'fresh_expert_control' not in r
 d=R/f'episode-{r["episode"]}';trace=[json.loads(x) for x in (d/'trace.jsonl').read_text().splitlines()];assert len(trace)==r['executed_frames'] and [x['frame'] for x in trace]==list(range(len(trace))) and trace[-1]['success'] and not any(x['success'] for x in trace[:-1]);assert r['first_success_frame']==trace[-1]['frame']
 assert r['initial_state_max_error']==0
 for cam,x in r['camera_errors'].items():assert x['intrinsic_max_error']==0 and x['rgb_mean_abs_error']<1;rgb.append(x['rgb_mean_abs_error'])
 v=d/'replay.mp4';j=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height','-of','json',str(v)],text=True));assert float(j['format']['duration'])>0 and j['streams'][0]['width']==320 and j['streams'][0]['height']==240
 out['successes']+=1;out[r['source_arm']]+=1;out['results'].append({'episode':r['episode'],'seed':r['seed'],'arm':r['source_arm'],'success':True,'source_frames':rec['frames'],'executed_frames':r['executed_frames'],'first_success_frame':r['first_success_frame'],'mean_command_tracking_error':r['mean_command_tracking_error'],'max_command_tracking_error':r['max_command_tracking_error'],'trace_sha256':hashlib.sha256((d/'trace.jsonl').read_bytes()).hexdigest(),'video':str(v)})
out['max_initial_rgb_mae_255scale']=max(rgb);out['initial_state_max_error']=0;out['all_intrinsics_equal']=True;out['plan_sha256']=hashlib.sha256((R/'plan.json').read_bytes()).hexdigest();assert out['left']==out['right']==5
(B/'status/expert-replay-verified.json').write_text(json.dumps(out,indent=2))
# The source archive already contains videos; stream them alongside the replay remotely.
sections=[]
with zipfile.ZipFile(B/'assets/robotwin-data/dataset/click_bell/demo_clean.zip') as z:
 for i in [0,1]:
  member=next(n for n in z.namelist() if n.endswith(f'/video/episode_{i:07d}.mp4'));dst=R/f'source-episode-{i}.mp4';dst.write_bytes(z.read(member));r=next(x for x in s['results'] if x['episode']==i)
  sections.append(f'<section><h2>seed {r["seed"]} / {r["source_arm"]} / 专家回放成功</h2><div><figure><figcaption>原始示范录像</figcaption><video controls preload="metadata" src="{dst.name}"></video></figure><figure><figcaption>原始动作 → 当前测评仿真</figcaption><video controls preload="metadata" src="episode-{i}/replay.mp4"></video></figure></div></section>')
(R/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>click_bell 专家轨迹回放</title><style>body{font-family:system-ui;margin:28px;background:#f5f6fa;color:#17223b}section{background:white;padding:20px;margin:20px 0;border-radius:12px}div{display:flex;gap:20px}figure{margin:0}video{width:480px;max-width:45vw}figcaption{padding:8px}</style><h1>原始专家动作回放：10/10 成功</h1><p>左右臂各5条；未调用模型。该数字是环境与控制链诊断，不是FLUX成功率。</p>'+''.join(sections))
PY
