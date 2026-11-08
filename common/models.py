"""Model loading helpers that work across transformers versions."""
from __future__ import annotations

import torch


def load_causal_lm(path_or_id: str, dtype: torch.dtype, revision: str | None = None, device: str = "cuda"):
    from transformers import AutoModelForCausalLM
    # device_map loads weights straight onto the GPU (no full copy staged in system RAM)
    kw = dict(revision=revision, low_cpu_mem_usage=True, device_map=device)
    try:
        model = AutoModelForCausalLM.from_pretrained(path_or_id, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path_or_id, torch_dtype=dtype, **kw)
    if model.dtype != dtype:
        model = model.to(dtype)
    return model.to(device)


def tiny_causal_lm(path_or_id: str, dtype: torch.dtype, device: str = "cuda", seed: int = 0):
    """Randomly initialised 2-layer model with the real tokenizer vocabulary (smoke tests only)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(path_or_id)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 32
    if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
        cfg.layer_types = cfg.layer_types[:2]
    if hasattr(cfg, "max_window_layers"):
        cfg.max_window_layers = 2
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg)
    return model.to(dtype).to(device)


def unique_named_parameters(model):
    """named_parameters() already de-duplicates tied tensors; kept explicit for clarity."""
    seen = set()
    for n, p in model.named_parameters(remove_duplicate=True):
        if id(p) in seen:
            continue
        seen.add(id(p))
        yield n, p
