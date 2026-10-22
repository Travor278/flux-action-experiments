R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
export PYTHONPATH="$R/flux-action-official-v1/src:$R/scripts" PYTHONNOUSERSITE=1
nohup "$R/policy-venv/bin/python" "$R/scripts/prepare_official_r2.py" cover_blocks > "$R/status/official-r2-cover_blocks-prepare.log" 2>&1 < /dev/null &
