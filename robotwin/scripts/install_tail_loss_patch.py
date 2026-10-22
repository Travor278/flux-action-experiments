"""Install opt-in temporal loss masks in an isolated FLUX source copy, never live eval code."""
from pathlib import Path
import sys
root=Path(sys.argv[1])
p=root/'src/flux_action/policy.py'
s=p.read_text()
assert 'action_time_mask: Tensor | None' not in s
s=s.replace('    event: Any = None\n', '    event: Any = None\n    action_time_mask: Tensor | None = None\n    video_time_mask: Tensor | None = None\n',1)
s=s.replace('        if self.action_mask is not None:\n            yield self.action_mask',
'''        if self.action_mask is not None:
            yield self.action_mask
        if self.action_time_mask is not None:
            yield self.action_time_mask
        if self.video_time_mask is not None:
            yield self.video_time_mask''',1)
old='        return PreparedWindows(idx_keep, latents, ctxs, state[idx_keep], actions[idx_keep], mask)'
new='''        result = PreparedWindows(idx_keep, latents, ctxs, state[idx_keep], actions[idx_keep], mask)
        if "action_time_mask" in batch:
            assert cfg.inference_profile != "history", "tail masks currently support single-observation packing"
            tm = torch.as_tensor(batch["action_time_mask"], device=device)[idx_keep]
            assert tm.shape == (len(idx_keep), cfg.chunk_size) and tm.dtype == torch.bool
            assert tm.any(-1).all(), "each sampled observation must have a real target"
            result.action_time_mask = tm
            count = torch.as_tensor(batch["video_valid_frames"], device=device)[idx_keep]
            assert ((count >= 1) & (count <= cfg.chunk_size + 1)).all()
            # Causal video VAE: future latent j ends at recorded frame 4*j.
            endpoints = torch.arange(1, latents.shape[2], device=device) * 4
            valid = endpoints[None, :] < count[:, None]
            result.video_time_mask = valid.repeat_interleave(latents.shape[-2]*latents.shape[-1], dim=1)
        return result'''
assert old in s;s=s.replace(old,new,1)
s=s.replace('            action_mask=prepared.action_mask,',
'''            action_mask=prepared.action_mask,
            action_time_mask=prepared.action_time_mask,
            video_time_mask=prepared.video_time_mask,''',1)
p.write_text(s)
p=root/'src/flux_action/processing/packing.py';s=p.read_text()
old='    action_mask: Tensor | None = None,\n    reduction: str = "joint_tokens",'
assert old in s;s=s.replace(old,'''    action_mask: Tensor | None = None,
    action_time_mask: Tensor | None = None,
    video_time_mask: Tensor | None = None,
    reduction: str = "joint_tokens",''',1)
old='    if reduction == "modalities":\n        total = video_weight * video.mean() + action_weight * weighted_action.mean()'
new='''    action_log, video_log = action, video
    if action_time_mask is not None or video_time_mask is not None:
        if action_time_mask is None or video_time_mask is None:
            raise ValueError("both temporal masks are required")
        am = action_time_mask.to(action.device, dtype=torch.bool).reshape_as(action)
        vm = video_time_mask.to(video.device, dtype=torch.bool).reshape_as(video)
        action_log = action.masked_select(am)
        video_log = video.masked_select(vm)
        weighted_action = weighted_action * am
        video = video * vm
        # Keep the original fixed token budget denominator: one short tail target
        # must not suddenly receive 32x the full-window loss weight.
        if reduction != "joint_tokens":
            raise ValueError("temporal masks require fixed-budget joint_tokens reduction")
    if reduction == "modalities":
        total = video_weight * video.mean() + action_weight * weighted_action.mean()'''
assert old in s;s=s.replace(old,new,1)
s=s.replace('return {"loss": total, "video_mse": video.mean().detach(), "action_mse": action.mean().detach()}',
'''return {"loss": total,
            "video_mse": (video_log.mean() if video_log.numel() else video.sum()*0).detach(),
            "action_mse": action_log.mean().detach()}''',1)
p.write_text(s)
q=root/'src/flux_action/training/trainer.py'; t=q.read_text()
old='''        if self.manifest is None:
            data["source_root"] = self.config.source_root'''
new='''        if self.manifest is None:
            data["source_root"] = self.config.source_root
            if self.config.dataset == "robotwin_tail_dataset:build":
                data["tail_manifest"] = json.loads((Path(self.config.index_dir) / "manifest.json").read_text())'''
assert old in t;t=t.replace(old,new,1);q.write_text(t)
print('TAIL_LOSS_PATCH_INSTALLED', root)
