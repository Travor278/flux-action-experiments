"""RoboDojo temporal target masks; preserve upstream fixed loss denominators."""
import torch
from . import packing


def latent_end_indices(num_frames):
    """Mirror the pretrained encoder's two parity-dependent TemporalMerging layers."""
    if not isinstance(num_frames,int) or num_frames < 1:
        raise ValueError('positive frame count required')
    ends = list(range(num_frames))
    for _ in range(2):
        if len(ends)%2:
            ends = [ends[0]]+ends
        ends = ends[1::2]
    return ends


def prepare_masks(batch, idx, cfg, latents):
    action = batch.get('action_time_mask')
    video = batch.get('video_valid_frames')
    if action is None and video is None:
        return None, None
    if action is None or video is None or cfg.inference_profile != 'history':
        raise ValueError('temporal masks require paired action/video masks and history profile')
    action = torch.as_tensor(action, device=latents.device)
    video = torch.as_tensor(video, device=latents.device)
    if action.dtype != torch.bool or video.dtype != torch.bool:
        raise ValueError('temporal masks must be boolean')
    if action.ndim != 2 or action.shape[1] != cfg.chunk_size or video.shape != (action.shape[0], cfg.n_obs_steps+cfg.chunk_size):
        raise ValueError('temporal mask shape mismatch')
    action, video = action[idx], video[idx, cfg.n_obs_steps:]
    # Each TemporalMerging prepends frame0 ONLY when its input length is odd.
    # Thus T32 endpoints are3,7,...31; T33 endpoints0,4,...32; T42 endpoints1,5,...41.
    # A fixed0,4,... mask is wrong for the independent history future clip.
    ends = torch.tensor(latent_end_indices(cfg.chunk_size),device=latents.device)
    valid = video.to(torch.int64).cumprod(1).bool()[:, ends]
    if latents.shape[2] - cfg.history_snapshots != valid.shape[1]:
        raise ValueError('unexpected VAE temporal packing; do not guess the mask')
    video_tokens = valid.repeat_interleave(latents.shape[-2]*latents.shape[-1], dim=1)
    return action, video_tokens


def flow_loss(pred, targets, modality, action_weight, video_weight, *, action_time_mask=None, video_token_mask=None, **kwargs):
    if action_time_mask is None and video_token_mask is None:
        return packing.flow_loss(pred, targets, modality, action_weight, video_weight, **kwargs)
    if action_time_mask is None or video_token_mask is None:
        raise ValueError('both temporal masks are required')
    p, t = dict(pred), dict(targets)
    for key, mask in [('x_'+modality, action_time_mask), ('x_video', video_token_mask)]:
        if mask.dtype != torch.bool or mask.ndim != 2 or mask.numel() != pred[key].shape[0]*pred[key].shape[1]:
            raise ValueError('token mask shape mismatch')
        if pred[key].shape != targets[key].shape:
            raise ValueError('prediction and target shape mismatch')
        mask = mask.reshape(pred[key].shape[:2]).unsqueeze(-1)
        p[key] = torch.where(mask, pred[key], torch.zeros_like(pred[key]))
        t[key] = torch.where(mask, targets[key], torch.zeros_like(targets[key]))
    # Keep token counts/denominators unchanged. A one-action tail must not be
    # amplified by K merely because K-1 synthetic targets were removed.
    return packing.flow_loss(p, t, modality, action_weight, video_weight, **kwargs)
