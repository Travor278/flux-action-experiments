B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,json,re,hashlib,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');E=B/'evaluation/handover_block/flux-tail-10k-test100';p=json.loads((B/'status/tail-handover-resume-plan.json').read_text());backup=B/'status/tail-handover-resume-evidence';raw=(E/'cases.jsonl').read_bytes();original=(backup/'cases.jsonl').read_bytes();assert raw.startswith(original)
assert hashlib.sha256((E/'protocol.json').read_bytes()).hexdigest()==p['prefix_sha256'][str(E/'protocol.json')]
text=re.sub(r'\x1b\[[0-9;]*m','',(B/'status/flux-tail-10k-test100-handover_block.log').read_text(errors='replace'));scores=[tuple(map(int,x)) for x in re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>.*?current seed:\s*(\d+)',text)]
assert len(scores)>=34 and scores[32]==(9,33,100039) and scores[33][1:]==(34,100040)
assert [x[1] for x in scores]==list(range(1,len(scores)+1)) and len({x[2] for x in scores})==len(scores)
prev=0
for record,(succ,n,seed) in zip(p['retained_pairs'],scores[:33]):
 assert record['seed']==seed and record['success']==bool(succ-prev);prev=succ
meta=json.loads((E/'resume-implementation.json').read_text());assert meta['expert_check'] is True and meta['total_test_num']==100
r={'time':time.time(),'complete':True,'prefix_33_bytes_and_scores_unchanged':True,'original_protocol_sha_unchanged':True,'resumed_seed100040_counted_once':True,'last_official':scores[-1],'scored_total':len(scores),'note':'Resume integrity only, not complete100 evaluation'}
(B/'status/tail-handover-resume-integrity-verified.json').write_text(json.dumps(r,indent=2))
PY
