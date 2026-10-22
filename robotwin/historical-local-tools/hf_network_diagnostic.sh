set -euo pipefail
python3 - <<'PY'
import socket,json,pathlib,time,ssl,os
r={'time':time.time(),'proxy_variables_present':{k:bool(os.environ.get(k)) for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']},'tests':[]}
for host in ['huggingface.co','hf.co','cas-server.xethub.hf.co']:
 ips=sorted({x[4][0] for x in socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)})
 t={'host':host,'addresses':ips}
 try:
  start=time.time()
  with socket.create_connection((host,443),timeout=5) as s:
   s.settimeout(5)
   with ssl.create_default_context().wrap_socket(s,server_hostname=host) as tls:t['tls']=tls.version()
  t['seconds']=time.time()-start;t['connected']=True
 except OSError as e:t.update(connected=False,error=str(e))
 r['tests'].append(t)
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');(B/'status/hf-network-diagnostic.json').write_text(json.dumps(r,indent=2))
PY
