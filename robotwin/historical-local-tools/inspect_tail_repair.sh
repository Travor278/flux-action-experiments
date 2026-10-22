B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,shutil
B=pathlib.Path.cwd();S=pathlib.Path(str(B).replace('/hdd/','/ssd/'));r={}
for t in ['click_bell','handover_block']:
 c=json.loads((B/f'configs/{t}-10k-opt-v1.json').read_text());r[t]={'config':c,'v3_exists':(B/f'assets/pi05-v3/{t}/meta/info.json').exists(),'stats':json.loads((B/f'assets/index/{t}/statistics.json').read_text())}
r['free_ssd']=shutil.disk_usage(S).free
(B/'status/tail-repair-input-inspection.json').write_text(json.dumps(r))
PY
