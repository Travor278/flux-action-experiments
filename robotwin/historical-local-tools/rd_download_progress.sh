set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,subprocess,time
R=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
r={'time':time.time(),'processes':[]}
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  s=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'inspect_robodojo_v3.py' in s or 'prepare_robodojo_remote.sh' in s:r['processes'].append({'pid':p.name,'cmd':s,'io':(p/'io').read_text()})
 except OSError:pass
r['files']=[{'path':str(p.relative_to(R)), 'size':p.stat().st_size} for p in (R/'official-source').rglob('*') if p.is_file() and '.cache' not in p.parts]
(R/'status/download-progress.json').write_text(json.dumps(r,indent=2))
PY