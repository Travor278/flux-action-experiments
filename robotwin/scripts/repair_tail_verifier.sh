set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,hashlib,shutil,time,os
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008'); p=B/'scripts/verify_tail_stage.py';gate=B/'status/tail-repair-ready.json';r=json.loads(gate.read_text());old=hashlib.sha256(p.read_bytes()).hexdigest();assert old==r['files']['scripts/verify_tail_stage.py']
shutil.copy2(p,B/'status/tail-verifier-before-logfix.py');shutil.copy2(gate,B/'status/tail-ready-before-logfix.json')
s=(B/'scripts/verify_tail_stage_repaired.py').read_bytes();compile(s,str(p),'exec');tmp=p.with_suffix('.new');tmp.write_bytes(s);tmp.replace(p)
for rel in ['scripts/verify_tail_stage.py','scripts/tail_log_records.py']:r['files'][rel]=hashlib.sha256((B/rel).read_bytes()).hexdigest()
r['verifier_logfix']={'time':time.time(),'old_sha':old,'reason':'valid JSON scalar NCCL output crashed dict-only audit; no model/data/training changes','regression_tests':2}
tmp=gate.with_suffix('.new');tmp.write_text(json.dumps(r,indent=2));tmp.replace(gate)
print('VERIFIER_REPAIRED',flush=True)
PY
PYTHONPATH="$B/flux-action-tail-v1/src:$B/scripts" "$B/flux-action/.venv/bin/python" "$B/scripts/verify_tail_stage.py" handover_block 5 > "$B/status/handover-stage5-repaired-audit.log" 2>&1
printf 'PASS\n' > "$B/status/tail-verifier-repair-complete.txt"
