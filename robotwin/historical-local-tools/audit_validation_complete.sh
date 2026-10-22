set -euo pipefail
python3 - <<'PY'
import pathlib,json,re,time,subprocess,math
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
out={'time':time.time(),'tasks':{},'pipeline_complete':(B/'status/flux-validation20-complete.txt').exists()}
for task in ['click_bell','handover_block']:
 E=B/f'evaluation/{task}/flux3000-validation20';result=json.loads((E/'result.json').read_text());pairs=json.loads((E/'paired_seeds.json').read_text());assert result['complete'] and len(pairs)==20 and sum(x['success'] for x in pairs)==result['successes']
 text=re.sub(r'\x1b\[[0-9;]*m','',(B/f'status/flux-{task}-validation20.log').read_text(errors='replace'))
 issues=[line[-500:] for line in text.replace('\r','\n').splitlines() if re.search(r'error|exception|skip|unstable|nan',line,re.I)]
 videos=list((B/f'RoboTwin/eval_result/{task}/FLUX_Action/demo_clean/flux3000-validation20').rglob('*.mp4'))
 n=20;s=result['successes'];z=1.96;p=s/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 out['tasks'][task]={'result':result,'wilson95':[center-delta,center+delta],'paired_seeds':[x['seed'] for x in pairs],'log_issues':issues[-20:],'video_count':len(videos),'video_samples':[{'path':str(v),'bytes':v.stat().st_size} for v in videos[:3]]}
out['own_processes']=[x for x in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if str(B) in x and ('setup_policy_server.py' in x or 'run_eval_recorded.py' in x)]
(B/'status/validation20-audit.json').write_text(json.dumps(out,indent=2))
PY
