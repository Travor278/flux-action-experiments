set -euo pipefail
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
mkdir -p "$R/system-libs/graphics-minimal"
for name in libGLU.so.1 libOpenGL.so.0 libGLdispatch.so.0; do cp -L "$B/system-libs/usr/lib/x86_64-linux-gnu/$name" "$R/system-libs/graphics-minimal/$name"; done
unset CUDA_VISIBLE_DEVICES VK_DRIVER_FILES
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH="$R/system-libs/graphics-minimal:${LD_LIBRARY_PATH:-}" OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
nohup bash -c 'R="$1"; timeout --signal=TERM --kill-after=15s 300s "$R/sim-venv/bin/python" "$R/scripts/render_smoke.py" > "$R/status/renderer-smoke-attempt3.log" 2>&1; echo "$?" > "$R/status/renderer-smoke-attempt3.exitcode"' bash "$R" > "$R/status/renderer-launch-attempt3.log" 2>&1 < /dev/null &
echo "$!" > "$R/status/renderer-smoke.pid"