set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
cd "$B"
PY="$B/flux-action/.venv/bin/python"
"$PY" -c 'import sys;sys.path.insert(0,"scripts");import profile,cProfile,benchmark_scaling,profile_worker;assert "scripts" not in profile.__file__;print("STDLIB_PROFILE_OK",profile.__file__)'
"$PY" scripts/make_configs.py
"$PY" scripts/verify_windows.py
