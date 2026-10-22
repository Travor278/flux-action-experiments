set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));out={'time':time.time(),'train':json.loads((B/'configs/click_bell-10k-opt-v1.json').read_text()),'export':json.loads((S/'runs/click_bell-10k-opt-v1/export-10000-ema/config.json').read_text()),'eval_protocol':json.loads((B/'evaluation/click_bell/flux-10k-test100/protocol.json').read_text()),'data_meta':json.loads((B/'assets/lerobot/click_bell/meta/info.json').read_text()),'index_files':[p.name for p in (B/'assets/index/click_bell').iterdir()]}
for p in (B/'assets/index/click_bell').glob('*.json'):
 j=json.loads(p.read_text());out[p.name]={'keys':list(j),'sample_episode':j.get('episodes',[None])[0]} if p.name=='manifest.json' else j
(B/'status/train-eval-audit-inputs.json').write_text(json.dumps(out,indent=2))
PY
