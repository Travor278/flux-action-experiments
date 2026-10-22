set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
if [ -f "$B/scripts/profile.py" ]; then mv "$B/scripts/profile.py" "$B/status/profile-shadowing-original.py.txt"; fi
python3 - <<'PY'
import pathlib,os,signal,json
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
found=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().split(b'\0')
  wanted={str(B/'scripts/profile_worker.py').encode(),str(B/'scripts/verify_windows.py').encode()}
  if wanted.intersection(args):
   found.append({'pid':int(p.parent.name),'cmd':b' '.join(args).decode(errors='replace')})
   os.kill(int(p.parent.name),signal.SIGTERM)
 except (OSError,ProcessLookupError):pass
(B/'status/profile-shadowing-cleanup.json').write_text(json.dumps(found,indent=2))
PY
if [ -d "$B/runs/profile" ]; then mv "$B/runs/profile" "$B/runs/invalid-profile-shadowing-20261008"; fi
if [ -f "$B/status/profile-results.json" ]; then mv "$B/status/profile-results.json" "$B/status/invalid-profile-results-shadowing.json"; fi
