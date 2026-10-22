from pathlib import Path
import hashlib,json,shutil,sys
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
task=sys.argv[1];assert task in ['cover_blocks','plug_in_charger']
g=json.loads((R/f'status/official-r2-{task}-ready.json').read_text());assert g['ready'] and g['fresh_base_optimizer']
for n,h in g['files'].items():
 with (R/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,n
cfg=json.loads((R/f'configs/{task}-10k-official-history-r2-20261021.json').read_text())
assert cfg['dataset']=='index' and 'prefix_memory_spec' not in cfg['policy']
assert cfg['policy']['n_obs_steps']==8 and cfg['policy']['history_snapshots']==2
idx=Path(cfg['index_dir']);m=json.loads((idx/'manifest.json').read_text())
with (idx/m['rows']['file']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['rows']['sha256']
assert shutil.disk_usage(Path(cfg['output_dir']).parent).free>300_000_000_000
print('OFFICIAL_R2_GATE_PASS',task,flush=True)
