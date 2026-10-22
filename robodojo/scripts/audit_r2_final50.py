"""Only this independent official-result/video audit publishes COMPLETE."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
task=sys.argv[1];assert task in ('cover_blocks','plug_in_charger')
out=R/'evaluation'/task/'flux-10k-official-history-r2-20261021-final50'
p=json.loads((out/'protocol.json').read_text());r=json.loads((out/'result.json').read_text())
assert (out/'EVALUATION_FINISHED.json').is_file() and p['training_step']==10000 and p['eval_num']==50
details=sorted(r['details'].items(),key=lambda x:int(x[0]));assert [int(k) for k,v in details]==list(range(50))
assert [v['layout_id'] for k,v in details]==p['layout_ids']==list(range(50))
success=sum(bool(v['success']) for k,v in details)
assert r['eval_time']==50 and abs(r['success_rate']-success/50)<1e-12
videos=[]
for key,detail in details:
 tag='success' if detail['success'] else 'fail'
 for camera in ['cam_high','cam_left_wrist','cam_right_wrist']:
  f=out/'official'/f'episode_{int(key):07d}_{camera}_{tag}.mp4'
  assert f.is_file() and f.stat().st_size>0,str(f)
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,duration,nb_frames','-of','json',str(f)],text=True))
  stream=info['streams'][0];assert stream['width']>0 and stream['height']>0 and float(stream['duration'])>0
  with f.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
  videos.append({'path':str(f),'bytes':f.stat().st_size,'sha256':digest,'stream':stream})
report={'time':time.time(),'verified':True,'task':task,'successes':success,'total':50,'success_rate':success/50,'unique_layouts':50,'videos':videos,'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),'result_sha256':hashlib.sha256((out/'result.json').read_bytes()).hexdigest()}
(out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='videos'}),flush=True)
