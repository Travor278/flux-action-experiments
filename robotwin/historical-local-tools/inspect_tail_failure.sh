set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
S=${B/\/hdd\//\/ssd\/}
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time,traceback
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));r={'time':time.time()}
for task in ['click_bell','handover_block']:
 p=S/f'runs/{task}-10k-tail-v1';r[task]={'train_tail':(p/'train.log').read_text(errors='replace')[-9000:],'metrics_tail':(p/'metrics.jsonl').read_text()[-6500:],'files':[x.name for x in p.iterdir()]}
(B/'status/tail-failure-investigation.json').write_text(json.dumps(r))
PY
