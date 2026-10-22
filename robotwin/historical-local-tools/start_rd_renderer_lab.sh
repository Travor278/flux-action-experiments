set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
unset CUDA_VISIBLE_DEVICES VK_DRIVER_FILES
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$R/xdg-cache" XDG_CONFIG_HOME="$R/xdg-config" XDG_DATA_HOME="$R/xdg-data"
nohup bash -c 'R="$1"; timeout --signal=TERM --kill-after=15s 180s "$R/sim-venv/bin/python" "$R/scripts/render_smoke_lab.py" > "$R/status/renderer-smoke-lab.log" 2>&1; echo "$?" > "$R/status/renderer-smoke-lab.exitcode"' bash "$R" > "$R/status/renderer-launch-lab.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/renderer-smoke.pid"