set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
exec 9> "$R/status/ws-deps-install.lock"
flock -n 9 || exit 3
exec > "$R/status/ws-deps-pip-install.log" 2>&1
set +e
timeout --signal=TERM --kill-after=5s 180s "$R/sim-venv/bin/python" -m pip --python "$R/policy-venv/bin/python" install --target "$R/ws-deps" --no-deps --disable-pip-version-check --retries 1 --timeout 20 pydantic==2.11.10 pydantic-core==2.33.2 annotated-types==0.8.0 typing-extensions==4.12.2 typing-inspection==0.4.2
code=$?
set -e
printf '%s\n' "$code" > "$R/status/ws-deps-pip-install.exitcode"
exit "$code"
