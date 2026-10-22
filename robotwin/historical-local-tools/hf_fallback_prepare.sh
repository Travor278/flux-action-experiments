set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,json,time,os,signal,shutil
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');S=pathlib.Path(str(B).replace('/hdd/','/ssd/'))
pid=int((B/'status/hf-final-upload.pid').read_text());p=pathlib.Path(f'/proc/{pid}')
if p.exists():
 cmd=(p/'cmdline').read_bytes().decode(errors='replace');assert str(B/'scripts/upload_flux_hf.py') in cmd
 (B/'status/hf-source-upload-process-before-stop.json').write_text(json.dumps({'time':time.time(),'pid':pid,'io':(p/'io').read_text(),'reason':'First ConnectTimeout and subsequent prolonged network wait; switch to authenticated browser delivery and local HF uploader'}))
 os.kill(pid,signal.SIGTERM)
if (B/'status/hf-final-upload.log').exists():shutil.copy2(B/'status/hf-final-upload.log',B/'status/hf-source-upload-attempt.log')
link=B/'assets/hf-publish-transfer'
if not link.exists():link.symlink_to(S/'hf-publish',target_is_directory=True)
assert link.resolve()==(S/'hf-publish').resolve()
proof=json.loads((B/'status/optimized-final-artifacts-verified.json').read_text())
for task in ['click_bell','handover_block']:
 dest=S/'hf-publish'/task;dest.mkdir(parents=True,exist_ok=True);source=S/f'runs/{task}-10k-opt-v1/export-10000-ema'
 for name in ['model.safetensors','config.json','manifest.json']:
  if not (dest/name).exists():os.link(source/name,dest/name)
  assert os.path.samefile(source/name,dest/name)
 if not (dest/'README.md').exists():
  s=(S/'hf-publish/click_bell/README.md').read_text().replace('click_bell',task).replace(str(round(proof['tasks']['click_bell']['final']['loss'],8)),str(round(proof['tasks'][task]['final']['loss'],8)))
  (dest/'README.md').write_text(s)
 (dest/'training_recipe.json').write_text((B/f'configs/{task}-10k-opt-v1.json').read_text())
PY
