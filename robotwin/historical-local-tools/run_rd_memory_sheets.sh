set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/policy-venv/bin/python" "$R/scripts/make_memory_contact_sheets.py" > "$R/status/memory-contact-sheets.log" 2>&1