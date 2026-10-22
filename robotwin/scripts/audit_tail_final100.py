import pathlib,json,time,re,hashlib,subprocess,sys
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
task=sys.argv[1];assert task in ['click_bell','handover_block'];label='flux-tail-10k-test100';E=B/f'evaluation/{task}/{label}'
result=json.loads((E/'result.json').read_text());pairs=json.loads((E/'paired_seeds.json').read_text());cfg=json.loads((E/'protocol.json').read_text())
assert result['complete'] and result['episodes']==len(pairs)==100 and result['split']=='test' and len({x['seed'] for x in pairs})==100
assert [x['episode'] for x in pairs]==list(range(1,101)) and sum(x['success'] for x in pairs)==result['successes']
assert cfg['seed']==0 and cfg['test_num']==100 and cfg['expert_check'] is True and cfg['task_config']=='demo_clean' and cfg['instruction_type']=='seen' and cfg['frequency']==15 and cfg['ckpt_setting']==label
log=re.sub(r'\x1b\[[0-9;]*m','',(B/f'status/{label}-{task}.log').read_text(errors='replace'));scores=[tuple(map(int,x)) for x in re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>.*?current seed:\s*(\d+)',log)]
assert len(scores)==100;prev=0
for i,((succ,total,seed),pair) in enumerate(zip(scores,pairs),1):
 assert total==i and seed==pair['seed'] and succ-prev==int(pair['success']);prev=succ
assert prev==result['successes']
V=B/f'RoboTwin/eval_result/{task}/FLUX_Action/demo_clean/{label}';videos=sorted(V.rglob('*.mp4'));assert len(videos)==100,len(videos)
video_checks=[]
for p in videos:
 assert p.stat().st_size>1000
 with p.open('rb') as f:assert b'ftyp' in f.read(32)
 a=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height:format=duration','-of','json',str(p)],text=True));assert a['streams'] and float(a['format']['duration'])>0
 video_checks.append({'name':p.name,'bytes':p.stat().st_size,'duration':float(a['format']['duration']),'stream':a['streams'][0]})
plan=json.loads((B/'status/flux-tail-final-test-plan.json').read_text());assert task+'-10k-tail-v1' in plan['tasks'][task]['checkpoint']
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in E.glob('*.json*')}
old=B/f'evaluation/{task}/flux-10k-test100/paired_seeds.json';comparison=None
if old.exists():
 oldpairs=json.loads(old.read_text());oldmap={x['seed']:x for x in oldpairs};common=[x for x in pairs if x['seed'] in oldmap]
 comparison={'old_complete_episodes':len(oldpairs),'same_seed_count':len(common),'same_prompt_count':sum(x['instruction']==oldmap[x['seed']]['instruction'] for x in common),'old_successes':sum(x['success'] for x in oldpairs)}
out={'time':time.time(),'complete':True,'task':task,'step':10000,'successes':prev,'episodes':100,'protocol_verified':True,'official_counts_unique_seeds_verified':True,'checkpoint':plan['tasks'][task]['checkpoint'],'files':files,'videos':video_checks,'video_count':100,'all_video_ffprobe_pass':True,'comparison':comparison}
(B/f'status/flux-tail-{task}-final100-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['videos','files']}))
