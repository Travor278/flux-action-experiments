R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/scripts:$R/flux-action/src" PYTHONNOUSERSITE=1
"$R/policy-venv/bin/python" "$R/scripts/install_resume_prefix_guard.py" > "$R/status/prefix-resume-contract.log" 2>&1
