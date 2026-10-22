"""Process-local helpers reused from the verified A2R pi05 workflow."""
import torch

def dataset_processors(config, stats, tokenizer_root):
    # Base-checkpoint processors belong to a newer LeRobot registry and describe
    # a different robot/action space. Build the stock 0.4.3 pipeline for this dataset.
    from lerobot.policies.pi05 import processor_pi05 as processors
    original_tokenizer = processors.TokenizerProcessorStep
    def local_tokenizer(*a, **kw):
        kw['tokenizer_name'] = str(tokenizer_root)
        return original_tokenizer(*a, **kw)
    processors.TokenizerProcessorStep = local_tokenizer
    try:
        return processors.make_pi05_pre_post_processors(config, dataset_stats=stats)
    finally:
        processors.TokenizerProcessorStep = original_tokenizer

def verify_token_embedding(policy, model_root):
    from safetensors import safe_open
    model = policy.model.paligemma_with_expert.paligemma
    embedding = model.get_input_embeddings().weight
    with safe_open(model_root / 'model.safetensors', framework='pt', device='cpu') as weights:
        names = [key for key in weights.keys() if 'paligemma_with_expert.paligemma.' in key and
                 (key.endswith('embed_tokens.weight') or key.endswith('lm_head.weight'))]
        assert names, 'Base checkpoint does not contain the token embedding or its tied output head'
        reference = weights.get_slice(names[0])[:4, :16].to(dtype=embedding.dtype)
    assert torch.equal(embedding[:4, :16].detach().cpu(), reference), 'Token embedding was not loaded from the base checkpoint'
    print('A2R_EMBEDDING base checkpoint sample verified', flush=True)
