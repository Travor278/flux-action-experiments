set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import os,pathlib,signal,time,json
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
pid=int((B/'status/curobo-bootstrap.pid').read_text())
p=pathlib.Path(f'/proc/{pid}/cmdline')
if p.exists():
 args=p.read_bytes().split(b'\0')
 if str(B/'scripts/bootstrap_curobo.sh').encode() not in args:raise RuntimeError('PID no longer belongs to old curobo watcher')
 os.kill(pid,signal.SIGTERM)
 time.sleep(1)
(B/'status/curobo-watcher-repair.json').write_text(json.dumps({'old_watcher_pid':pid,'reason':'replace circular wait on task imports with basic dependency gate'}))
PY
"$B/sim-venv/bin/python" - <<'PY' > "$B/status/sim-dependencies-ready.tmp"
import torch,numpy,sapien,mplib
print('SIM_DEPENDENCIES_OK',torch.__version__,numpy.__version__)
PY
mv "$B/status/sim-dependencies-ready.tmp" "$B/status/sim-dependencies-ready.txt"
nohup bash "$B/scripts/bootstrap_curobo_v2.sh" > "$B/status/curobo-bootstrap-v2.log" 2>&1 < /dev/null &
echo $! > "$B/status/curobo-bootstrap.pid"
