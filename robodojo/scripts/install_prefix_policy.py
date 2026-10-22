"""One-time, SHA-guarded experiment prefix-conditioning integration."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

R = Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
repo=R/'flux-action'; base=repo/'src/flux_action'
receipt=R/'status/prefix-policy-integration.json'
if receipt.exists(): raise SystemExit('Already integrated; inspect receipt, do not reapply')
sha=lambda x:hashlib.sha256(x).hexdigest()
policy=(base/'policy.py').read_bytes();config=(base/'config.py').read_bytes()
assert sha(policy)==json.loads((R/'status/temporal-mask-integration.json').read_text())['policy_patched_sha256']
assert config==subprocess.check_output(['git','-C',str(repo),'show','HEAD:src/flux_action/config.py'])
def change(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)
c=config.decode()
c=change(c,'    condition_on_past_actions: bool = False\n','    condition_on_past_actions: bool = False\n    prefix_memory_spec: dict | None = None  # experiment-owned causal RGB memory contract\n')
c=change(c,'        self.camera_keys = tuple(self.camera_keys)\n','''        if self.prefix_memory_spec is not None:
            from .processing.prefix_memory import PrefixMemorySpec
            spec = PrefixMemorySpec.from_dict(self.prefix_memory_spec)
            if self.inference_profile != "history" or self.fps != spec.fps:
                raise ValueError("prefix memory requires history profile at its native fps")
            if self.augment or self.camera_dropout:
                raise ValueError("prefix v1 requires shared unaugmented, non-dropped RGB")
            self.prefix_memory_spec = spec.to_dict()
        self.camera_keys = tuple(self.camera_keys)
''')
p=policy.decode()
p=change(p,'from .processing import history, normalization, packing, temporal_masking','from .processing import history, normalization, packing, temporal_masking, prefix_memory')
p=change(p,'    video_token_mask: Tensor | None = None\n','    video_token_mask: Tensor | None = None\n    prefix_tokens: Tensor | None = None\n    prefix_ids: Tensor | None = None\n')
p=change(p,'            yield self.video_token_mask\n','            yield self.video_token_mask\n        if self.prefix_tokens is not None:\n            yield self.prefix_tokens\n            yield self.prefix_ids\n')
p=change(p,'        self._observation_history = history.ObservationHistory(self.config.n_obs_steps)\n','''        self._observation_history = history.ObservationHistory(self.config.n_obs_steps)
        self._prefix_memory = (prefix_memory.PrefixMemoryBank(self.config.camera_keys, prefix_memory.PrefixMemorySpec.from_dict(self.config.prefix_memory_spec)) if self.config.prefix_memory_spec is not None else None)
''')
old='        return PreparedWindows(idx_keep, latents, ctxs, state[idx_keep], actions[idx_keep], mask, action_time_mask=action_time_mask, video_token_mask=video_token_mask)'
new='''        prefix_tokens, prefix_ids = prefix_memory.encode_prefix(self.video_vae, batch, cfg, idx_keep, device)
        return PreparedWindows(idx_keep, latents, ctxs, state[idx_keep], actions[idx_keep], mask, action_time_mask=action_time_mask, video_token_mask=video_token_mask, prefix_tokens=prefix_tokens, prefix_ids=prefix_ids)'''
p=change(p,old,new)
p=change(p,'            video = history.pack_video(prepared.latents, cfg)\n','            video = prefix_memory.merge_prefix(history.pack_video(prepared.latents, cfg), prepared.prefix_tokens, prepared.prefix_ids)\n')
p=change(p,'''                        history.pack_conditioning(
                            self.video_vae, cams[i], state[i : i + 1], commands[i : i + 1], self.config
                        ),''','''                        prefix_memory.inference_conditioning(
                            self.video_vae, cams[i], state[i : i + 1], commands[i : i + 1], self.config, batch, i
                        ),''')
p=change(p,'            window = self._observation_history.append(batch, self.config, self._last_command)\n','''            if self._prefix_memory is not None:
                batch = {**batch, **self._prefix_memory.append(batch)}
            window = self._observation_history.append(batch, self.config, self._last_command)
''')
for name,text in [('policy.py',p),('config.py',c)]: compile(text,name,'exec')
module=(R/'scripts/prefix_memory.py').read_bytes();compile(module,'prefix_memory.py','exec')
backup=R/'status/prefix-policy-original';backup.mkdir(exist_ok=False)
(backup/'policy.py').write_bytes(policy);(backup/'config.py').write_bytes(config)
(base/'processing/prefix_memory.py').write_bytes(module)
for name,text in [('policy.py',p),('config.py',c)]:
    tmp=base/(name+'.tmp');tmp.write_text(text);tmp.replace(base/name)
receipt.write_text(json.dumps({'time':time.time(),'original_policy_sha256':sha(policy),'original_config_sha256':sha(config),'patched_policy_sha256':sha((base/'policy.py').read_bytes()),'patched_config_sha256':sha((base/'config.py').read_bytes()),'module_sha256':sha(module),'scope':'Experimental prefix conditioning, defaultNone preserves original profile; tests/GPU acceptance pending'},indent=2))
print('PREFIX_POLICY_SOURCE_INTEGRATED',flush=True)
