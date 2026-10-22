"""Bounded exact memoization for frozen deterministic VAE prefix encoding.

No training config, image, timestamp, sample order, or RNG changes. Cache only
identical input bytes, invalidate when frozen parameter versions change.
"""
from collections import OrderedDict
import hashlib
import torch

class ExactPrefixVAECache:
    def __init__(self,vae,max_bytes=1024**3):
        self.vae=vae;self.max_bytes=max_bytes;self.cache=OrderedDict()
        self.bytes=0;self.hits=0;self.misses=0;self.versions=None
        assert all(not p.requires_grad for p in vae.module.parameters())
    @torch.no_grad()
    def encode_task(self,video):
        versions=tuple((p.data_ptr(),p._version,p.dtype,p.device) for p in self.vae.module.parameters())
        if versions!=self.versions:
            self.cache.clear();self.bytes=0;self.versions=versions
        raw=video.detach().contiguous().cpu()
        key=(tuple(raw.shape),str(raw.dtype),hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).digest())
        if key in self.cache:
            self.hits+=1;value=self.cache.pop(key);self.cache[key]=value
            return value.to(device=video.device,copy=True)
        self.misses+=1
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state(video.device) if video.is_cuda else None
        result=self.vae.encode_task(video)
        assert torch.equal(cpu_rng,torch.get_rng_state()),'VAE consumes CPU RNG; caching forbidden'
        if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(video.device)),'VAE consumes CUDA RNG; caching forbidden'
        stored=result.detach().cpu().clone();size=stored.numel()*stored.element_size()
        if size<=self.max_bytes:
            while self.bytes+size>self.max_bytes:
                _,old=self.cache.popitem(last=False);self.bytes-=old.numel()*old.element_size()
            self.cache[key]=stored;self.bytes+=size
        return result
