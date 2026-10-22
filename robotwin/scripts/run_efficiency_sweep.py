"""Bounded 2-H100 tuning, selecting by complete optimizer-update wall throughput."""
import os,json,pathlib,subprocess,time,statistics,math,hashlib,csv,datetime,shutil,signal
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));ROOT=S/'runs/two-gpu-efficiency-sweep-v1'
PY=B/'flux-action/.venv/bin/python';WORKER=B/'scripts/train_efficiency_worker.py'
def invoke(cfg,out,env,overrides=(),log='train.log'):
 out.mkdir(exist_ok=True,parents=True)
 with (out/log).open('w') as f:
  proc=subprocess.Popen([str(PY),'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(WORKER),str(cfg),*overrides],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:return proc.wait(timeout=1200)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   return 124
def jsonrows(path):
 return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
def assess(out,rc):
 rows=jsonrows(out/'metrics.jsonl');updates=[r for r in rows if 'step_time' in r]
 result={'returncode':rc,'updates':len(updates),'valid':rc==0 and len(updates)==12 and all(math.isfinite(r[k]) for r in updates for k in ['loss','grad_norm','step_time'])}
 if not result['valid']:
  log=(out/'train.log').read_text(errors='replace');result['oom']='out of memory' in log.lower();result['tail']=log[-2000:];return result
 steady=[r for r in updates if 5<=r['step']<=12]
 rate=statistics.median(b['time']-a['time'] for a,b in zip(steady,steady[1:]))
 result.update(seconds_per_update=rate,windows_per_second=32/rate,peak_gpu_gb=max(r['peak_mem_gb'] for r in updates),data_wait_median=statistics.median(r['data_wait'] for r in steady),steady_start=steady[0]['time'],steady_end=steady[-1]['time'])
 return result
def main():
 proof=json.loads((B/'status/two-gpu-perf-repair-result.json').read_text());assert proof['candidate_valid'] and proof['save_resume_export']
 assert not ROOT.exists(),'Never overwrite an existing sweep'
 ROOT.mkdir();base=json.loads((B/'configs/two-gpu-perf-repair.json').read_text())
 subprocess.run([str(PY),str(B/'scripts/test_async_ema.py')],check=True)
 tests=[('buffered_pf2_t4',1,2,4,'buffered'),('buffered_pf16_t4',1,16,4,'buffered'),('buffered_pf16_t8',1,16,8,'buffered'),('async_pf16_t4',1,16,4,'async'),('async_pf16_t8',1,16,8,'async'),('async_micro2_pf16_t4',2,16,4,'async')]
 report={'started':time.time(),'complete':False,'results':[]}
 for name,micro,prefetch,threads,mode in tests:
  out=ROOT/name;cfg={**base,'output_dir':str(out),'steps':12,'checkpoint_every':12,'resume':'none','windows_per_rank':micro,'grad_accumulation':32//(2*micro),'prefetch_factor':prefetch,'log_every':1}
  path=B/f'configs/efficiency-{name}.json';path.write_text(json.dumps(cfg,indent=2))
  env={**os.environ,'CPU_EMA_MODE':mode,'CPU_THREADS':str(threads),'OMP_NUM_THREADS':str(threads),'BENCH_NO_SAVE':'1'}
  out.mkdir();g=(out/'gpu.csv').open('w');mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=g)
  report['current']=name;(B/'status/efficiency-sweep.json').write_text(json.dumps(report,indent=2))
  try:rc=invoke(path,out,env)
  finally:mon.terminate();mon.wait();g.close()
  result={'name':name,'micro':micro,'prefetch':prefetch,'cpu_threads':threads,'ema_mode':mode,'config':str(path),**assess(out,rc)}
  if result['valid']:
   telemetry=[]
   for row in csv.reader((out/'gpu.csv').read_text().splitlines()):
    try:
     stamp=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp()
     if result['steady_start']<=stamp<=result['steady_end']:telemetry.append((int(row[1]),float(row[2]),float(row[3])))
    except (ValueError,IndexError):pass
   result['gpu_util_mean']={str(i):statistics.mean(x[1] for x in telemetry if x[0]==i) for i in set(x[0] for x in telemetry)}
  report['results'].append(result);(B/'status/efficiency-sweep.json').write_text(json.dumps(report,indent=2));print('SWEEP_RESULT '+json.dumps(result),flush=True)
  if name=='async_micro2_pf16_t4' and result['valid']:
   tests.append(('async_micro4_pf16_t4',4,16,4,'async'))
 valid=[r for r in report['results'] if r['valid']];assert valid,'No valid configuration'
 best=min(valid,key=lambda r:r['seconds_per_update']);report['selected']=best;report['current']='selected_save_resume_test'
 (B/'status/efficiency-sweep.json').write_text(json.dumps(report,indent=2))
 cfg=json.loads(pathlib.Path(best['config']).read_text());out=ROOT/'selected-validation';cfg['output_dir']=str(out);cfg['steps']=3;cfg['checkpoint_every']=3
 path=B/'configs/efficiency-selected-validation.json';path.write_text(json.dumps(cfg,indent=2))
 env={**os.environ,'CPU_EMA_MODE':best['ema_mode'],'CPU_THREADS':str(best['cpu_threads']),'OMP_NUM_THREADS':str(best['cpu_threads'])};env.pop('BENCH_NO_SAVE',None)
 assert invoke(path,out,env)==0 and (out/'step-3/COMPLETE').exists()
 assert invoke(path,out,env,['steps=4','resume=auto'],log='resume.log')==0 and (out/'step-4/COMPLETE').exists()
 assert any(r.get('event')=='resume' and r['step']==3 for r in jsonrows(out/'metrics.jsonl'))
 # Test safe manual-stop at an update boundary using the same worker/checkpoint path.
 (out/'STOP_AFTER_UPDATE').touch()
 assert invoke(path,out,env,['steps=8','resume=auto'],log='safe-stop.log')==0
 stopped=json.loads((out/'STOPPED_SAFELY.json').read_text());assert stopped['step']==5 and (out/'step-5/COMPLETE').exists()
 (out/'STOP_AFTER_UPDATE').unlink()
 with (out/'export.log').open('w') as f:subprocess.run([str(PY),'-m','flux_action.cli','export-checkpoint','--checkpoint',str(out/'step-5'),'--output',str(out/'export-5-ema'),'--profile','ema_0p10','--dtype','bfloat16'],check=True,stdout=f,stderr=subprocess.STDOUT,env=env)
 m=json.loads((out/'export-5-ema/manifest.json').read_text())
 for n,h in m['sha256'].items():
  with (out/'export-5-ema'/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
 report.update(complete=True,current=None,save_resume_export=True,safe_stop_verified=True,completed_at=time.time(),validation_run=str(out))
 (B/'status/efficiency-sweep.json').write_text(json.dumps(report,indent=2));(B/'status/efficiency-selected-verified.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
