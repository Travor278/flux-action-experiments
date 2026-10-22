B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time,csv,statistics,math,shutil,os
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));m=json.loads((B/'status/tail-training-monitor.json').read_text());now=time.time();r={'time':now,'source_monitor_time':m['time'],'tasks':{}}
assert now-m['time']<300
for task,a in m['tasks'].items():
 assert not a.get('error') and a['ranks']==[0,1] and a['last_update']['step']>220
 u=a['last_update'];assert all(math.isfinite(u[k]) for k in ['loss','grad_norm'])
 rows=list(csv.reader((S/f'runs/{task}-10k-tail-v1/gpu.csv').read_text().splitlines()))[-120:]
 gpu={i:{'mean_util':statistics.mean(float(x[2]) for x in rows if x[1].strip()==i),'latest_memory_mib':[float(x[3]) for x in rows if x[1].strip()==i][-1]} for i in ['0','1']}
 remaining=max(0,u['time']+(10000-u['step'])*a['seconds_per_update']-now)+1800
 r['tasks'][task]={'step':u['step'],'loss':u['loss'],'grad_norm':u['grad_norm'],'seconds_per_update':a['seconds_per_update'],'eta_seconds_including_30min_save_export_allowance':remaining,'gpu_recent_5min':gpu}
r['next_minutes']=math.ceil(max(a['eta_seconds_including_30min_save_export_allowance'] for a in r['tasks'].values())/60)
r['estimated_complete_unix']=now+r['next_minutes']*60;r['free_ssd']=shutil.disk_usage(S).free
r['cleanup_candidates']=[str(p) for root in [B/'assets',S] for p in root.iterdir() if p.is_file() and p.suffix in ['.partial','.part','.tar','.tmp']]
r['old_eval_pids_alive']={str(p):pathlib.Path(f'/proc/{p}').exists() for p in [4049716,4076687]}
(B/'status/tail-full-training-eta.json').write_text(json.dumps(r,indent=2))
PY
