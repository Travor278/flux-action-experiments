set -euo pipefail
python3 - <<'PY'
import pathlib,json,time,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');a=B/'status/flux-click-10k-final100-audit.json';r=json.loads(a.read_text());root=pathlib.Path(r['video_root']);vs=sorted(root.rglob('episode*.mp4'));assert len(vs)==100 and len({v.parent for v in vs})==1
assert {v.name for v in vs}=={f'episode{i}.mp4' for i in range(100)}
for v in vs:
 assert v.stat().st_size>1000
 with v.open('rb') as f:assert b'ftyp' in f.read(128)
pairs=json.loads((B/'evaluation/click_bell/flux-10k-test100/paired_seeds.json').read_text());ids=[p['episode']-1 for p in pairs if p['success']]+[0,99];probes=[]
for i in sorted(set(ids)):
 v=vs[0].parent/f'episode{i}.mp4';j=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height','-of','json',str(v)],text=True));assert float(j['format']['duration'])>0 and j['streams'][0]['width']>0
 probes.append({'episode':i,'seed':pairs[i]['seed'],'success':pairs[i]['success'],'probe':j})
r['video_headers_all100_verified']=True;r['video_samples_probed']=probes;r['media_audit_time']=time.time();a.write_text(json.dumps(r,indent=2));(B/'status/flux-click-video-audit-summary.json').write_text(json.dumps({'all100_headers':True,'probed_episodes':[p['episode'] for p in probes],'successes':r['successes'],'episodes':100}))
PY
