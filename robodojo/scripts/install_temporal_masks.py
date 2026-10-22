"""Apply one recorded integration to this experiment's idle FLUX source only."""
from pathlib import Path
import hashlib
import json
import time
import subprocess

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
target = R/'flux-action/src/flux_action/policy.py'
marker = R/'status/temporal-mask-integration.json'
digest = lambda b: hashlib.sha256(b).hexdigest()
if marker.exists():
    raise SystemExit('Already integrated; inspect receipt instead of reapplying')
raw = target.read_bytes()
repo = R/'flux-action'
head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
assert head == '830f1ce178d7aea52f18fde31d8e8e36ce6bae21', head
assert raw == subprocess.check_output(['git', '-C', str(repo), 'show', 'HEAD:src/flux_action/policy.py'])
src = raw.decode()
changes = [
    ('from .processing import history, normalization, packing', 'from .processing import history, normalization, packing, temporal_masking'),
    ('    event: Any = None\n', '    event: Any = None\n    action_time_mask: Tensor | None = None\n    video_token_mask: Tensor | None = None\n'),
    ('            yield self.action_mask\n', '            yield self.action_mask\n        if self.action_time_mask is not None:\n            yield self.action_time_mask\n        if self.video_token_mask is not None:\n            yield self.video_token_mask\n'),
    ('        return PreparedWindows(idx_keep, latents, ctxs, state[idx_keep], actions[idx_keep], mask)', '        action_time_mask, video_token_mask = temporal_masking.prepare_masks(batch, idx_keep, cfg, latents)\n        return PreparedWindows(idx_keep, latents, ctxs, state[idx_keep], actions[idx_keep], mask, action_time_mask=action_time_mask, video_token_mask=video_token_mask)'),
    ('        losses = packing.flow_loss(\n', '        losses = temporal_masking.flow_loss(\n'),
    ('            action_mask=prepared.action_mask,\n', '            action_mask=prepared.action_mask,\n            action_time_mask=prepared.action_time_mask,\n            video_token_mask=prepared.video_token_mask,\n'),
]
for before, after in changes:
    assert src.count(before) == 1, before
    src = src.replace(before, after)
compile(src, str(target), 'exec')
backup = R/'status/temporal-mask-original-policy.py'
assert not backup.exists()
backup.write_bytes(raw)
module = (R/'scripts/temporal_masking.py').read_bytes()
dest = target.parent/'processing/temporal_masking.py'
assert not dest.exists()
compile(module, str(dest), 'exec')
dest.write_bytes(module)
temp = target.with_suffix('.py.tmp')
temp.write_text(src)
temp.replace(target)
marker.write_text(json.dumps({'time':time.time(), 'policy_original_sha256':digest(raw), 'policy_patched_sha256':digest(target.read_bytes()), 'mask_module_sha256':digest(module), 'scope':'Experiment-specific policy prepare/forward/loss only; upstream original retained. No trainer dataset wiring or GPU acceptance yet.'}, indent=2))
print('TEMPORAL_MASK_SOURCE_INTEGRATED', flush=True)
