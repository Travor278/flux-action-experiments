B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,time
R=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013');out=R/'status/transfer-io-only.json'
def snap():
 r={'time':time.time(),'net':pathlib.Path('/proc/net/dev').read_text(),'pids':{}}
 for pid in [17663,17679]:
  p=pathlib.Path(f'/proc/{pid}');r['pids'][str(pid)]={'io':(p/'io').read_text(),'wchan':(p/'wchan').read_text()} if p.exists() else None
 return r
r={'first':snap()};out.write_text(json.dumps(r));time.sleep(20);r['second']=snap();out.write_text(json.dumps(r))
PY