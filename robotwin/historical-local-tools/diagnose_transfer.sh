set -eu
B=/workspace/hdd/codex_remote_ops/flux_piperx_20261008
python3 - <<'PY'
import pathlib,os,json,re,time
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
out={'time':time.time(),'safe_flags':{k:os.environ.get(k) for k in ['HF_HUB_DISABLE_XET','HF_XET_HIGH_PERFORMANCE','HF_XET_NUM_CONCURRENT_RANGE_GETS']},'logs':{}}
for p in sorted(pathlib.Path('/tmp/flux-piperx-xet').rglob('*.log'))[-2:]:
 lines=p.read_text(errors='replace').splitlines()[-12:]
 out['logs'][p.name]=[re.sub(r'https?://[^\s"<>]+','<URL omitted>',x)[:1600] for x in lines]
for folder in ['assets/.robotwin-data.download']:
 out[folder]=[{'path':str(p.relative_to(B)),'bytes':p.stat().st_size,'modified':p.stat().st_mtime} for p in (B/folder).rglob('*') if p.is_file() and p.stat().st_size>1000000]
(B/'status/transfer-diagnosis.json').write_text(json.dumps(out,indent=2))
PY
