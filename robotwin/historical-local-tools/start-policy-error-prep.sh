set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
python3 - <<'PY'
import pathlib
p=pathlib.Path('scripts/launch_policy_error_audit.sh');p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
PY
bash -n scripts/launch_policy_error_audit.sh
nohup "$B/flux-action/.venv/bin/python" scripts/prepare_policy_error_audit.py > status/policy-error-audit-prepare.log 2>&1 < /dev/null &
echo "$!" > status/policy-error-audit-prepare.pid
