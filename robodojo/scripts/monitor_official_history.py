from pathlib import Path
import json,time,statistics,shutil
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
S=Path(str(R).replace('/hdd/','/ssd/'));result={'time':time.time(),'free_ssd_bytes':shutil.disk_usage(S).free,'tasks':{}}
for task in ('cover_blocks','match_and_pick_from_conveyor'):
    out=S/'runs'/f'{task}-10k-official-history-v1';record={'exists':out.exists(),'output':str(out)};rows=[]
    p=out/'metrics.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(r,dict):rows.append(r)
    updates=[r for r in rows if 'step_time' in r];record['latest']=updates[-1] if updates else None
    record['resume_events']=[r for r in rows if r.get('event')=='resume'][-4:]
    cutoff=max((r.get('time',0) for r in record['resume_events']),default=0)
    restart=R/'status/official-history-restart-audit-20261015.json'
    if restart.exists():
        restart_record=json.loads(restart.read_text());cutoff=max(cutoff,restart_record['time'])
    fresh=[r for r in updates if r.get('time',0)>=cutoff]
    record['last_recorded_update']=record['latest'];record['fresh_update_count']=len(fresh)
    record['awaiting_fresh_updates']=not fresh
    if fresh:record['latest']=fresh[-1]
    elif restart.exists():
        record['latest']={'step':restart_record['tasks'][task]['resume_step'],'time':cutoff,'event':'waiting_for_fresh_update'}
    stable=[r for r in fresh if r['step']>=220][-31:];rates=[]
    for a,b in zip(stable,stable[1:]):
        if b['step']>a['step'] and a['step']//1000==b['step']//1000 and b.get('time',0)>a.get('time',0):rates.append((b['time']-a['time'])/(b['step']-a['step']))
    record['stable_seconds_per_update']=statistics.median(rates) if rates else None
    record['checkpoint_markers']=[str(p.relative_to(out)) for p in out.glob('step-*/COMPLETE')]
    for marker in ('INITIAL_CHECKPOINT_VERIFIED.json','EARLY_CHECKPOINT_VERIFIED.json','STOPPED_SAFELY.json','TRAIN_COMPLETE.json'):
        p=out/marker;record[marker]=json.loads(p.read_text()) if p.exists() else None
    for name in ('initial.log','early.log','train.log','launch.log','export.log','gpu.csv'):
        p=out/name;record[name]=p.read_text(errors='replace')[-4000:] if p.exists() else None
    result['tasks'][task]=record
(R/'status/official-history-monitor.json').write_text(json.dumps(result,indent=2));print(json.dumps({'time':result['time'],'tasks':{k:v['latest'] for k,v in result['tasks'].items()}}))
