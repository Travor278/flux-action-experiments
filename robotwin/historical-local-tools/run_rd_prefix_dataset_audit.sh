set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
cd "$R/scripts"
"$R/policy-venv/bin/python" audit_prefix_dataset.py > "$R/status/prefix-dataset-real-audit.log" 2>&1