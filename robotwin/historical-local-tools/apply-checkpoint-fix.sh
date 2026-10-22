set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
T="$B/flux-action/src/flux_action/training/checkpoint.py"
test -f "$T"
test ! -f "$B/status/checkpoint-original.py" && cp "$T" "$B/status/checkpoint-original.py"
cp "$B/scripts/checkpoint-fixed.py" "$T.new"
mv "$T.new" "$T"
export PYTHONPATH="$B/flux-action/src:$B/scripts" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2
nohup "$B/flux-action/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=2 "$B/scripts/test_checkpoint_cpu.py" > "$B/status/checkpoint-cpu-test.log" 2>&1 < /dev/null &
echo $! > "$B/status/checkpoint-cpu-test.pid"
