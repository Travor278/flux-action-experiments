set -euo pipefail
python3 - <<'PY'
import pathlib,json,time,re,hashlib,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));task='click_bell';E=B/f'evaluation/{task}/flux-10k-test100';r=json.loads((E/'result.json').read_text());p=json.loads((E/'paired_seeds.json').read_text());cfg=json.loads((E/'protocol.json').read_text())
assert r['complete'] and r['episodes']==len(p)==100 and r['split']=='test' and len({x['seed'] for x in p})==100
assert [x['episode'] for x in p]==list(range(1,101)) and sum(x['success'] for x in p)==r['successes']
assert all(x['seed']>=100000 and x['action_type']=='joint' and x['instruction'] for x in p)
assert cfg['seed']==0 and cfg['test_num']==100 and cfg['expert_check'] is True and cfg['task_config']=='demo_clean' and cfg['instruction_type']=='seen' and cfg['frequency']==15 and cfg['ckpt_setting']=='flux-10k-test100'
log=re.sub(r'\x1b\[[0-9;]*m','',(B/f'status/flux-10k-test100-{task}.log').read_text(errors='replace'));scores=[tuple(map(int,m)) for m in re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>.*?current seed:\s*(\d+)',log)]
assert len(scores)==100
prev=0
for i,((succ,total,seed),pair) in enumerate(zip(scores,p),1):
 assert total==i and seed==pair['seed'] and succ-prev==int(pair['success']);prev=succ
assert prev==r['successes']
V=B/f'RoboTwin/eval_result/{task}/FLUX_Action/demo_clean/flux-10k-test100';videos=sorted(V.rglob('*.mp4')) if V.exists() else []
files={f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in E.glob('*.json*')}
ho=B/'evaluation/handover_block/flux-10k-test100/protocol.json';h=json.loads(ho.read_text());assert h['test_num']==100 and h['seed']==0 and h['expert_check'] is True and h['ckpt_setting']=='flux-10k-test100'
out={'time':time.time(),'task':task,'step':10000,'complete':True,'successes':r['successes'],'episodes':100,'official_counts_and_unique_seeds_verified':True,'protocol_verified':True,'files':files,'video_root':str(V),'video_count':len(videos),'video_samples':[{'path':str(f),'bytes':f.stat().st_size} for f in videos[:8]],'handover_protocol_verified':True,'disk_free_bytes':shutil.disk_usage(S).free}
(B/'status/flux-click-10k-final100-audit.json').write_text(json.dumps(out,indent=2))
PY
