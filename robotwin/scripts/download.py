import hashlib
import json
import os
import pathlib
import time
import zipfile
from huggingface_hub import HfApi, snapshot_download

B = pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
os.environ.setdefault('HF_XET_CACHE', '/tmp/flux-piperx-xet')
api = HfApi()
specs = [
    ('TianxingChen/RoboTwin2.0', 'dataset', ['dataset/click_bell/demo_clean.zip', 'dataset/handover_block/demo_clean.zip'], 'robotwin-data'),
    ('black-forest-labs/flux-3-action-base', 'model', ['flux-3-action-base.safetensors', 'video_vae.safetensors', 'text_encoder/*', 'LICENSE.md', 'README.md'], 'flux-base'),
]
for repo, kind, patterns, name in specs:
    manifest_path = B / 'assets' / (name + '.manifest.json')
    final = B / 'assets' / name
    if final.exists() and manifest_path.exists():
        print('Already verified', name, flush=True)
        continue
    info = api.repo_info(repo, repo_type=kind, files_metadata=True)
    tmp = B / 'assets' / ('.' + name + '.download')
    print('DOWNLOAD', repo, info.sha, flush=True)
    snapshot_download(repo, repo_type=kind, revision=info.sha, allow_patterns=patterns, local_dir=tmp, max_workers=4)
    expected = {f.rfilename: f for f in info.siblings}
    files = []
    for f in sorted(tmp.rglob('*')):
        if not f.is_file() or '.cache' in f.relative_to(tmp).parts:
            continue
        rel = f.relative_to(tmp).as_posix()
        meta = expected[rel]
        size = f.stat().st_size
        if meta.size is not None and size != meta.size:
            raise ValueError(('size mismatch', rel, size, meta.size))
        digest = hashlib.file_digest(f.open('rb'), 'sha256').hexdigest()
        if meta.lfs and digest != meta.lfs.sha256:
            raise ValueError(('hash mismatch', rel))
        if f.suffix == '.zip':
            with zipfile.ZipFile(f) as z:
                bad = z.testzip()
                if bad:
                    raise ValueError(('bad archive', bad))
        if f.suffix == '.safetensors':
            from safetensors import safe_open
            with safe_open(f, framework='numpy') as sf:
                if not sf.keys():
                    raise ValueError(('empty tensors', rel))
        files.append({'path': rel, 'bytes': size, 'sha256': digest})
    tmp.rename(final)
    manifest_path.write_text(json.dumps({'repo': repo, 'revision': info.sha, 'verified_at': time.time(), 'files': files}, indent=2))
    print('VERIFIED', name, sum(f['bytes'] for f in files), flush=True)
