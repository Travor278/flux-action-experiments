set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
G=/workspace/hdd/codex_remote_ops/A2R_pickup_gen_20261007
if [ ! -d "$B/system-libs/usr" ]; then
 mkdir -p "$B/system-libs"
 cp -a "$G/system-libs/usr" "$B/system-libs/"
 cp "$G/system-libs/nvidia_icd.json" "$B/system-libs/nvidia_icd.json"
fi
cd "$B/RoboTwin"
"$B/sim-venv/bin/python" scripts/update_embodiment_config_path.py > "$B/status/embodiment-paths.log" 2>&1
python3 - <<'PY'
import pathlib,hashlib,json
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
records=[]
for p in sorted((B/'system-libs').rglob('*')):
 if p.is_symlink():records.append({'path':str(p.relative_to(B)),'link':str(p.readlink())})
 elif p.is_file():
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()
  records.append({'path':str(p.relative_to(B)),'bytes':p.stat().st_size,'sha256':h})
(B/'status/graphics-libraries-manifest.json').write_text(json.dumps({'source':'read-only copy of existing A2R Ubuntu user-space graphics libraries','files':records},indent=2))
PY
