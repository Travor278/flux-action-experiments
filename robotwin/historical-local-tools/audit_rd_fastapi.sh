R=/workspace/hdd/codex_remote_ops/flux_robodojo_20261013
"$R/sim-venv/bin/python" - <<'PY' > "$R/status/fastapi-dependency-audit.json"
import importlib.metadata as m,json
print(json.dumps({d.metadata['Name']:[x for x in d.requires or [] if 'fastapi' in x.lower()] for d in m.distributions() if any('fastapi' in x.lower() for x in d.requires or [])}))
PY