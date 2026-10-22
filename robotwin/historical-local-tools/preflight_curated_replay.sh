set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY' > "$B/status/curated-replay-preflight.json"
import ast,pathlib,json,shutil,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');E=B/'evaluation/handover_block/flux-opt-10000-validation20';ref=json.loads((B/'evaluation/handover_block/flux3000-validation20/paired_seeds.json').read_text());log=B/'status/flux-opt-10000-validation20-handover_block.log'
module=ast.parse((B/'scripts/run_eval_curated.py').read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='scored_pairs')
import re
namespace={'log':log,'out':E,'expected':{x['seed']:x['instruction'] for x in ref},'re':re,'json':json}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<scored-pairs-real-partial-audit>','exec'),namespace)
existing=namespace['scored_pairs']();pending=[x['seed'] for x in ref if x['seed'] not in existing];assert len(existing)==18 and pending==[900003,900015]
source=(B/'RoboTwin/scripts/eval_policy_xpolicylab.py').read_text();assert source.count('    while succ_seed < test_num:\n        render_freq = args["render_freq"]')==1
evidence=B/'status/curated-replay-first-failure';evidence.mkdir(exist_ok=True)
for p in [log,E/'cases.jsonl',E/'protocol.json']:
 if p.exists():shutil.copy2(p,evidence/p.name)
print(json.dumps({'time':time.time(),'verified':True,'retained_scored_episodes':18,'missing_seeds':pending,'original_loop_matched':True,'failure_evidence':str(evidence)}))
PY
bash -n "$B/scripts/launch_flux_milestone_validation_v2.sh"
"$B/flux-action/.venv/bin/python" -m py_compile "$B/scripts/run_eval_curated.py"
