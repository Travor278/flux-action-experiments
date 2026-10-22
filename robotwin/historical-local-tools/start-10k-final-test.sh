set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,hashlib,json,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');p=B/'scripts/launch_flux_10k_test100.sh';p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
(B/'status/flux-10k-launcher-source.json').write_text(json.dumps({'time':time.time(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'10k only, 100 test episodes per task'}))
PY
bash -n "$B/scripts/launch_flux_10k_test100.sh"
nohup bash "$B/scripts/launch_flux_10k_test100.sh" > "$B/status/flux-10k-final-test-launch.log" 2>&1 < /dev/null &
echo "$!" > "$B/status/flux-final-test.pid"
