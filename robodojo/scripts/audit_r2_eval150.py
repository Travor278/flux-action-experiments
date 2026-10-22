from pathlib import Path
import hashlib,json,subprocess,sys,time,os
from eval150_contract import output,validate_result
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
task=sys.argv[1];seed=int(os.environ['FLUX_EVAL_SEED']);out=output(R,task,seed)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
protocol=json.loads((out/'protocol.json').read_text());result=json.loads((out/'result.json').read_text())
plan=json.loads((R/'status/r2-eval150-plan.json').read_text());spec=plan['tasks'][task]['seeds'][str(seed)]
assert (out/'EVALUATION_FINISHED.json').exists() and protocol['seed']==seed and protocol['training_step']==10000
assert protocol['plan_sha256']==sha(R/'status/r2-eval150-plan.json')
assert protocol['layout_sha256']==spec['layout_sha256'] and protocol['layout_ids']==spec['layout_ids']
assert json.loads((out/'official/_result.json').read_text())==result
metrics=validate_result(result,spec['layout_ids'])
resume=json.loads((out/'resume.json').read_text());assert resume['success_nums']==metrics['successes'] and resume['fail_nums']==50-metrics['successes']
assert resume['details']==result['details'] and abs(resume['total_score']*2-metrics['score'])<1e-8
assert not resume.get('abandoned_layout_ids'), 'Unstable layouts need diagnosis, not silent replacement'
effective=json.loads((out/'effective-config.json').read_text());assert effective['core_sha256']==spec['effective_config_sha256'] and effective['assembled_from_official_ast']
videos=[]
for key,detail in sorted(result['details'].items(),key=lambda x:int(x[0])):
 for camera in ['cam_high','cam_left_wrist','cam_right_wrist']:
  tag='success' if detail['success'] else 'fail';p=out/'official'/f'episode_{int(key):07d}_{camera}_{tag}.mp4'
  assert p.is_file() and p.stat().st_size>0
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,duration,nb_frames','-of','json',str(p)],text=True))['streams'][0]
  assert info['width']>0 and info['height']>0 and float(info['duration'])>0
  videos.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'stream':info})
report={'time':time.time(),'verified':True,'task':task,'eval_seed':seed,**metrics,'model_sha256':protocol['checkpoint_sha256']['model.safetensors'],'plan_sha256':protocol['plan_sha256'],'protocol_sha256':sha(out/'protocol.json'),'result_sha256':sha(out/'result.json'),'videos':videos}
(out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='videos'}),flush=True)
