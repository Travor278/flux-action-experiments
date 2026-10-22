B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,json,time,fcntl,subprocess,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
r={'time':time.time(),'complete':False,'tasks':{},'locks_free':{},'model_payloads_transferred_to_operator':False,'hf_contains':'Previous uncorrected10k checkpoints only; new models not published'}
assert (B/'status/flux-tail-test-complete.txt').is_file()
for t in ['click_bell','handover_block']:
 a=json.loads((B/f'status/flux-tail-{t}-final100-audit.json').read_text());result=json.loads((B/f'evaluation/{t}/flux-tail-10k-test100/result.json').read_text());assert a['complete'] and a['episodes']==100 and a['video_count']==100 and a['all_video_ffprobe_pass'] and a['successes']==result['successes']
 r['tasks'][t]={'successes':a['successes'],'episodes':100,'rate':a['successes']/100,'checkpoint':a['checkpoint'],'metadata_sha':a['files'],'all100_videos_verified':True,'comparison':a.get('comparison')}
p=json.loads((B/'status/tail-handover-resume-plan.json').read_text());pairs=json.loads((B/'evaluation/handover_block/flux-tail-10k-test100/paired_seeds.json').read_text())
for old,new in zip(p['retained_pairs'],pairs[:33],strict=True):
 for key in ['episode','seed','instruction','success','action_type']:assert old[key]==new[key],(key,old,new)
r['handover_retained_first33_exact']=True
for name in ['flux-final-test.lock','flux-milestone-validation.lock']:
 with (B/'status'/name).open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);r['locks_free'][name]=True
procs=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if str(B) in cmd and any(x in cmd for x in ['setup_policy_server.py','run_eval_tail_resume.py','launch_tail_handover_resume.sh','launch_flux_tail_test100.sh']):procs.append({'pid':p.name,'cmd':cmd})
assert not procs,procs
r['own_eval_processes_remaining']=procs;r['gpu_after']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
r['free_ssd']=shutil.disk_usage(S).free;r['cleanup_candidates']=[str(p) for root in [B/'assets',S] for p in root.iterdir() if p.is_file() and p.suffix in ['.tmp','.part','.partial','.tar']]
r['complete']=True;(B/'status/flux-tail-experiment-delivered.json').write_text(json.dumps(r,indent=2));print('DELIVERY_AUDIT_PASS')
PY
