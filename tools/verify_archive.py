"""Offline source-only archive checks. Does not launch training or access remote hosts."""
from pathlib import Path
import ast,hashlib,json,re,sys
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'docs/SOURCE_MANIFEST.json').read_text())
for record in manifest['files']:
    p=root/record['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==record['public_sha256'],record['path']
for p in root.rglob('*.py'):
    if '.git' not in p.parts and '__pycache__' not in p.parts:ast.parse(p.read_text(encoding='utf8'),filename=str(p))
patterns=[r'hf_[A-Za-z0-9]{20,}',r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
for p in root.rglob('*'):
    if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts:continue
    assert p.suffix not in ('.safetensors','.distcp','.pt','.pth','.mp4','.parquet','.npy'),p
    text=p.read_text(encoding='utf8')
    for pattern in patterns:assert not re.search(pattern,text),f'Potential credential: {p.relative_to(root)}'
print(f'PASS: {len(manifest["files"])} source provenance records; Python syntax; payload/credential-pattern checks')
