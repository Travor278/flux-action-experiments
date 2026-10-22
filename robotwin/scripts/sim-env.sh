# Source this only in the experiment's simulation process.
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="$B/system-libs/nvidia_icd.json"
export XDG_RUNTIME_DIR=/tmp/flux-piperx-sim-runtime
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export PYTHONPATH="$B/RoboTwin:$B/XPolicyLab:$B:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=/tmp/flux-piperx-sim-torch
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
