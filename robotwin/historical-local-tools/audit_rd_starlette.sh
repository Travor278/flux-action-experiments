R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/sim-venv/bin/python" - <<'PY' > "$R/status/starlette-dependency-audit.json"
import importlib.metadata as m,json
out={}
for d in m.distributions():
 req=[x for x in d.requires or [] if 'starlette' in x.lower()]
 if req:out[d.metadata['Name']]=req
print(json.dumps(out))
PY