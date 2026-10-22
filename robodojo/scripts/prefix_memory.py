"""Experimental fixed-tick visual memory; no privileged task inputs.

This module is a prototype contract/buffer. Policy token wiring and a matching
export loader are required before training or deployment can enable it.
"""
from dataclasses import dataclass
import json
import torch


@dataclass(frozen=True)
class PrefixMemorySpec:
    ticks: tuple = (0,25,50)
    fps: int = 25
    version: str = 'causal_prefix_rgb_v1'

    def __post_init__(self):
        object.__setattr__(self,'ticks',tuple(self.ticks))
        if self.version != 'causal_prefix_rgb_v1' or self.fps != 25:
            raise ValueError('unsupported memory contract')
        if not self.ticks or self.ticks[0] != 0 or tuple(sorted(set(self.ticks))) != self.ticks:
            raise ValueError('anchors must be unique sorted ticks beginning at zero')
        if any(isinstance(x,bool) or not isinstance(x,int) or x < 0 for x in self.ticks):
            raise ValueError('invalid anchor tick')

    def indices(self, current_tick):
        if isinstance(current_tick,bool) or not isinstance(current_tick,int) or current_tick < 0:
            raise ValueError('invalid episode tick')
        # An anchor whose capture time has not arrived uses CURRENT observation,
        # never future data; this exact rule is shared by offline and online.
        return tuple(min(t,current_tick) for t in self.ticks)

    def to_dict(self):
        return {'version':self.version,'ticks':list(self.ticks),'fps':self.fps,'unavailable_rule':'current_observation','reset':'every_episode'}

    @classmethod
    def from_dict(cls, data):
        if set(data) != {'version','ticks','fps','unavailable_rule','reset'} or data['unavailable_rule'] != 'current_observation' or data['reset'] != 'every_episode':
            raise ValueError('incompatible memory serialization')
        return cls(tuple(data['ticks']),data['fps'],data['version'])


class PrefixMemoryBank:
    """One environment; append ON EVERY control tick, including queued actions."""
    def __init__(self, camera_keys, spec):
        self.keys = tuple(camera_keys)
        self.spec = spec
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError('invalid camera keys')
        self.reset()

    def reset(self):
        self.tick = -1
        self.anchors = {}
        self.current = None

    def append(self, images):
        current = {}
        for k in self.keys:
            image = images[k]
            if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
                raise ValueError('single-environment uint8 BCHW RGB required')
            if self.current is not None and (image.shape != self.current[k].shape or image.device != self.current[k].device):
                raise ValueError('camera shape/device changed without reset')
            current[k] = image.detach().clone()
        self.tick += 1
        self.current = current
        if self.tick in self.spec.ticks:
            self.anchors[self.tick] = current
        indices = self.spec.indices(self.tick)
        result = {'prefix_tick_indices':torch.tensor([indices],dtype=torch.long,device=current[self.keys[0]].device),'episode_tick':torch.tensor([self.tick],dtype=torch.long,device=current[self.keys[0]].device)}
        for k in self.keys:
            result['prefix.'+k] = torch.stack([(self.anchors[t] if t in self.anchors else current)[k] for t in indices],dim=1)
        return result


def prefix_relative_seconds(indices, episode_tick, spec):
    if indices.ndim != 2 or episode_tick.shape != (indices.shape[0],):
        raise ValueError('memory timestamp shape mismatch')
    expected = torch.tensor([spec.indices(int(t)) for t in episode_tick.tolist()],device=indices.device)
    if not torch.equal(indices,expected):
        raise ValueError('future or incompatible prefix timestamps')
    return (indices-episode_tick[:,None]).float()/spec.fps


@torch.no_grad()
def encode_prefix(vae, batch, cfg, indices, device):
    """Shared training/inference encoding, with a fixed visual-memory time origin."""
    from . import packing
    from ..models.positional import batched_prc_vid, times_to_ids
    if cfg.prefix_memory_spec is None:
        return None, None
    spec = PrefixMemorySpec.from_dict(cfg.prefix_memory_spec)
    ticks = torch.as_tensor(batch['prefix_tick_indices'],device=device)[indices]
    current = torch.as_tensor(batch['episode_tick'],device=device)[indices]
    times = prefix_relative_seconds(ticks,current,spec)
    # Existing history's most recent image lies at this visual time coordinate.
    times = times + (cfg.n_obs_steps-1)/cfg.video_position_fps
    images = []
    for key in cfg.camera_keys:
        raw = torch.as_tensor(batch['prefix.'+key])[indices]
        if raw.dtype != torch.uint8 or raw.ndim != 5 or raw.shape[1:3] != (len(spec.ticks),3):
            raise ValueError('prefix images must be uint8 B,A,3,H,W')
        valid = batch.get('prefix.'+key+'_valid_mask')
        if valid is not None and not torch.as_tensor(valid)[indices].all():
            raise ValueError('missing real prefix camera frame; diagnose source before training')
        images.append(raw)
    cams = torch.stack(images,dim=1)
    tokens, positions = [], []
    for i in range(len(indices)):
        canvas = packing.materialize_video(cams[i].cpu().float()/255,None,device,layout=cfg.camera_layout,canvas_hw=cfg.canvas_hw)
        per_tokens, per_ids = [], []
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.device(device).type=='cuda'):
            for j in range(len(spec.ticks)):
                latent = vae.encode_task(canvas[None,:,j:j+1].to(torch.bfloat16)).clone()[...,:cfg.latent_hw[0],:cfg.latent_hw[1]]
                token, ids = batched_prc_vid(latent,times_to_ids(times[i:i+1,j:j+1]))
                per_tokens.append(token);per_ids.append(ids)
        tokens.append(torch.cat(per_tokens,dim=1));positions.append(torch.cat(per_ids,dim=1))
    return torch.cat(tokens),torch.cat(positions)


def merge_prefix(conditioning, tokens, positions):
    if tokens is None:
        return conditioning
    return {**conditioning,'x_video_cond':torch.cat([tokens,conditioning['x_video_cond']],dim=1),'x_video_cond_ids':torch.cat([positions,conditioning['x_video_cond_ids']],dim=1)}


def inference_conditioning(vae, cameras, states, commands, cfg, batch, index):
    from . import history
    regular = history.pack_conditioning(vae,cameras,states,commands,cfg)
    tokens, positions = encode_prefix(vae,batch,cfg,[index],states.device)
    return merge_prefix(regular,tokens,positions)
