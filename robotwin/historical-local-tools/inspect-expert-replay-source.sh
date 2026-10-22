set -euo pipefail
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
"$B/flux-action/.venv/bin/python" - <<'PY'
import pathlib,json,zipfile,h5py,io,time,subprocess
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008');out={'time':time.time(),'tasks':{}}
for task in ['click_bell','handover_block']:
 with zipfile.ZipFile(B/f'assets/robotwin-data/dataset/{task}/demo_clean.zip') as z:
  names=z.namelist();n=sorted(n for n in names if n.endswith('.hdf5'))[0]
  with h5py.File(io.BytesIO(z.read(n))) as h:
   ds={}
   def visit(name,obj):
    if isinstance(obj,h5py.Dataset) and not name.endswith('/colors'):ds[name]={'shape':list(obj.shape),'dtype':str(obj.dtype)}
   h.visititems(visit)
   v=h['instructions'][()];ins=json.loads(v.decode());out['tasks'][task]={'auxiliary_files':[n for n in names if not n.endswith('.hdf5') and not n.endswith('/')],'attrs':{k:str(v) for k,v in h.attrs.items()},'datasets':ds,'instructions_type':type(ins).__name__,'instructions_keys':list(ins) if isinstance(ins,dict) else None,'sample_file':n,'initial_state':{k:h['state'][k][0].tolist() for k in h['state']},'head_extrinsic':h['vision/cam_head/extrinsics_matrix'][0].tolist()}
  for n in names:
   if ('seed' in n.lower() or n.endswith('.json')) and z.getinfo(n).file_size<200000:out['tasks'][task].setdefault('aux_contents',{})[n]=z.read(n).decode(errors='replace')
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
(B/'status/expert-replay-source-inspection.json').write_text(json.dumps(out,indent=2))
PY
