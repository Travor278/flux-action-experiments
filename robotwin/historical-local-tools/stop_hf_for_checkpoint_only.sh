set -euo pipefail
python3 - <<'PY'
import pathlib,os,signal,json,time,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');pid=int((B/'status/hf-final-upload.pid').read_text());p=pathlib.Path(f'/proc/{pid}')
if p.exists():
 cmd=(p/'cmdline').read_bytes().decode(errors='replace');assert str(B/'scripts/upload_flux_hf.py') in cmd
 os.kill(pid,signal.SIGTERM)
for name in ['hf-final-upload.log','hf-final-uploads.json']:
 f=B/'status'/name
 if f.exists():shutil.copy2(f,B/'status'/('hf-before-checkpoint-only-'+name))
(B/'status/hf-checkpoint-only-scope.json').write_text(json.dumps({'time':time.time(),'reason':'User restricted uploaded content to FLUX checkpoint files only; preserve source and Xet resume cache','files':['model.safetensors','config.json','manifest.json'],'no_local_model_staging':True,'pi05_eval_cancelled':True},indent=2))
PY
