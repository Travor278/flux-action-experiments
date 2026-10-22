"""Delete only the individually inspected files from this task's prohibited staging."""
import pathlib,json,time
base=pathlib.Path('C:/Users/USER/.codex/transfer').resolve()
files={
 'flux-pi05-eval-20261010/click_bell/model.safetensors':660914176,
 'flux-pi05-eval-20261010/click_bell/config.json':2327,
 'flux-pi05-eval-20261010/click_bell/policy_postprocessor_step_0_unnormalizer_processor.safetensors':9416,
 'flux-pi05-eval-20261010/click_bell/policy_postprocessor.json':663,
 'flux-pi05-eval-20261010/click_bell/policy_preprocessor_step_2_normalizer_processor.safetensors':9416,
 'flux-pi05-eval-20261010/click_bell/policy_preprocessor.json':1977,
 'flux-pi05-eval-20261010/click_bell/train_config.json':6587,
 'flux-final-hf-20261010/click_bell/config.json':3001,
}
deleted=[]
for relative,size in files.items():
 p=base/relative;assert p.resolve().is_relative_to(base) and not p.is_symlink()
 if p.exists():
  assert p.is_file() and p.stat().st_size==size,'Inspected file changed; do not remove blindly'
  p.unlink();deleted.append({'path':str(p),'bytes':size})
for rel in ['flux-pi05-eval-20261010/click_bell','flux-pi05-eval-20261010/handover_block','flux-pi05-eval-20261010','flux-final-hf-20261010/click_bell','flux-final-hf-20261010/handover_block','flux-final-hf-20261010']:
 p=base/rel
 if p.exists():assert not any(p.iterdir());p.rmdir()
record={'time':time.time(),'reason':'User forbids local model staging; stopped all task transfer writers first','deleted':deleted}
(pathlib.Path(__file__).parent/'prohibited-local-transfer-cleanup.json').write_text(json.dumps(record,indent=2));print(json.dumps({'files_removed':len(deleted),'bytes_removed':sum(x['bytes'] for x in deleted)}))
