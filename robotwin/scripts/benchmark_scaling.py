def main():
    import csv,json,math,os,pathlib,signal,statistics,subprocess,sys,time
    B=pathlib.Path(__file__).resolve().parents[1]
    import torch
    if torch.cuda.device_count() != 8 or any('H100' not in torch.cuda.get_device_name(i) for i in range(8)):
        raise RuntimeError('Scaling benchmark requires the authorized 8-H100 allocation')
    for gate in ['status/runtime-ready.txt','status/data-ready.txt','assets/flux-base.manifest.json']:
        if not (B/gate).is_file(): raise RuntimeError('Missing preparation gate: '+gate)
    base=json.loads((B/'configs/handover_block.json').read_text())
    results=[]
    for n in [2,4,8]:
        for micro in [1,2]:
            tag=f'{n}gpu-micro{micro}'
            out=B/'runs/profile'/tag
            out.mkdir(parents=True,exist_ok=False)
            cfg={**base,'output_dir':str(out),'steps':12,'cooldown_start':None,'cooldown_steps':0,
                 'frozen_steps':0,'trunk_warmup_steps':1,'heads_warmup_steps':1,'resume':'none',
                 'shard_size':n,'windows_per_rank':micro,'grad_accumulation':32//(n*micro),'log_every':1}
            config_path=out/'config.json'; config_path.write_text(json.dumps(cfg,indent=2))
            env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,range(n))),'OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={n}',str(B/'scripts/profile_worker.py'),str(config_path)]
            telemetry=(out/'gpu.csv').open('w')
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=telemetry)
            began=time.time(); timed_out=False
            (B/'status/profile-progress.json').write_text(json.dumps({'phase':'running_candidate','tag':tag,'gpus':n,'micro_batch':micro,'started':began,'target_steps':12,'global_batch':32},indent=2))
            with (out/'train.log').open('w') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try: rc=proc.wait(timeout=1200)
                except subprocess.TimeoutExpired:
                    timed_out=True; os.killpg(proc.pid,signal.SIGTERM)
                    try: rc=proc.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); rc=proc.wait()
            monitor.terminate(); monitor.wait(); telemetry.close()
            rows=[]
            if (out/'metrics.jsonl').exists():
                rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
                rows=[r for r in rows if 'step_time' in r and r['step']>=5]
            valid=rc==0 and len(rows)>=8 and all(math.isfinite(r[k]) for r in rows for k in ['loss','grad_norm','step_time'])
            result={'tag':tag,'gpus':n,'micro_batch':micro,'global_batch':32,'returncode':rc,'timeout':timed_out,'valid':valid,'elapsed':time.time()-began}
            if valid:
                # Include data stalls by measuring intervals between wall-clock metric timestamps.
                intervals=[rows[i]['time']-rows[i-1]['time'] for i in range(1,len(rows))]
                step=statistics.median(intervals)
                samples=[];by_gpu={i:[] for i in range(n)}
                for row in csv.reader((out/'gpu.csv').open()):
                    if len(row)==5 and int(row[1])<n:
                        # Match the same steady-state interval, not model loading.
                        import datetime
                        ts=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp()
                        if rows[0]['time']<=ts<=rows[-1]['time']:
                            samples.append(float(row[2]));by_gpu[int(row[1])].append(float(row[2]))
                result.update(step_seconds=step,windows_per_second=32/step,windows_per_second_per_gpu=32/step/n,
                              peak_mem_gb=max(r['peak_mem_gb'] for r in rows),gpu_utilization_mean=statistics.mean(samples) if samples else None,
                              training_eta_seconds=3000*step,telemetry_samples=len(samples),
                              per_gpu_utilization_mean={i:statistics.mean(v) if v else None for i,v in by_gpu.items()})
            else:
                log_text=(out/'train.log').read_text(errors='replace')
                result['error_tail']=log_text[-3000:]
                result['oom']='out of memory' in log_text.lower() or 'OutOfMemoryError' in log_text
                lines=log_text.splitlines();error_index=next((i for i,line in enumerate(lines) if 'Traceback' in line or 'Error:' in line),0)
                result['first_error_context']='\n'.join(lines[max(0,error_index-3):error_index+30])
            results.append(result)
            (B/'status/profile-results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(result),flush=True)
            if not valid:
                if result['oom'] or timed_out: break
                raise RuntimeError('Non-resource benchmark failure; stop and diagnose '+tag)
    valid=[r for r in results if r['valid']]
    if not valid: raise SystemExit('No valid full-training profile')
    fastest=max(valid,key=lambda r:r['windows_per_second'])
    efficient=max(valid,key=lambda r:r['windows_per_second_per_gpu'])
    utilized=max(valid,key=lambda r:r['gpu_utilization_mean'] if r['gpu_utilization_mean'] is not None else -1)
    # Prefer efficiency for two simultaneous task runs; retain the absolute fastest separately.
    decision={'fastest':fastest,'most_efficient':efficient,'most_utilized':utilized,'selected':efficient,
              'reason':'maximize aggregate throughput per allocated GPU for the two independent training tasks',
              'scope':'2/4/8 H100, microbatch 1/2, fixed global batch 32; 4 warmup + 8 measured updates; checkpoint I/O excluded'}
    (B/'status/profile-decision.json').write_text(json.dumps(decision,indent=2))
    (B/'status/profile-progress.json').write_text(json.dumps({'phase':'complete','finished':time.time(),'selected':efficient['tag']},indent=2))

if __name__ == "__main__":
    main()
