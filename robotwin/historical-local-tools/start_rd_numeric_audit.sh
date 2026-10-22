R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
nohup "$R/policy-venv/bin/python" "$R/scripts/audit_numeric.py" > "$R/status/numeric-audit.log" 2>&1 < /dev/null &