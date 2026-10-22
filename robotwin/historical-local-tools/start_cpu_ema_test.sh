B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
export PYTHONPATH="$B/flux-action/src:$B/scripts" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2
nohup "$B/flux-action/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=2 "$B/scripts/test_cpu_power_ema.py" > "$B/status/cpu-ema-unit.log" 2>&1 < /dev/null &
echo $! > "$B/status/cpu-ema-unit.pid"
