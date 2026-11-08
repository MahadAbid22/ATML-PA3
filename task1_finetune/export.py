"""Export evaluation models for Task 1.

* Full FT / L2-SP: FP32 master weights at the required update -> FP16 Hugging Face model.
* LoRA / OPLoRA: exported factors (projected A~, B~ for OPLoRA) are merged into a fresh
  copy of the original backbone (W = W0 + (alpha/r) B~A~, merged in FP32, stored FP16).
  Merged vs unmerged logits are compared on a small fixed batch.

Usage:  python -m task1_finetune.export --condition lora_r16 [--smoke]
"""
from __future__ import annotations

import argparse

import torch

from common.lora import inject_adapters, load_adapter_factors, merge_factors_into
from common.models import load_causal_lm, tiny_causal_lm, unique_named_parameters
from common.utils import OUTPUTS, RESULTS, ROOT, load_yaml, log, save_json
from task1_finetune.train import CFG, run_dir

CHECK_TEXTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "A man is sitting on a roof. He starts pulling up roofing on the roof.",
    "Is the sky blue on a clear day? Answer yes or no.",
    "Which of these is a renewable resource: coal, oil, wind, or natural gas?",
]


def condition_source(cond: str, cfg: dict) -> tuple[str, int]:
    if cond == "full_ft_mid":
        return "full_ft", cfg["train"]["midpoint_update"]
    return cond, cfg["train"]["updates"]


def model_dir(cond: str, smoke: bool = False):
    return OUTPUTS / "task1" / ("smoke_models" if smoke else "models") / cond


def _base(cfg, dtype, smoke, kind="full"):
    if smoke and kind == "full":
        return tiny_causal_lm(cfg["model"], dtype, "cuda", seed=cfg["seed"])
    return load_causal_lm(cfg["model"], dtype, cfg["model_revision"], "cuda")


@torch.no_grad()
def _logits(model, tok, texts):
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    out = model(**enc).logits.float()
    return out, enc["attention_mask"].bool()


def export(cond: str, smoke: bool = False) -> dict:
    from transformers import AutoTokenizer
    cfg = load_yaml(CFG)
    method, update = condition_source(cond, cfg)
    if smoke:  # smoke runs: 4 updates, midpoint at 2
        update = 2 if cond == "full_ft_mid" else 4
    mcfg = cfg["methods"][method]
    rdir = run_dir(method, smoke)
    out = model_dir(cond, smoke)
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    info = {"condition": cond, "method": method, "update": update}

    if mcfg["kind"] == "full":
        model = _base(cfg, torch.float32, smoke)
        sd = torch.load(rdir / f"ckpt_{update}" / "model.pt", map_location="cpu")
        for n, p in unique_named_parameters(model):
            p.data.copy_(sd[n])
        model = model.half()
    else:
        factors = torch.load(rdir / "adapter_export.pt", map_location="cpu")
        scaling = mcfg["alpha"] / mcfg["r"]
        # unmerged reference: FP32 backbone + exported factors as plain adapters (no dropout)
        ref = _base(cfg, torch.float32, smoke, mcfg["kind"])
        inject_adapters(ref, cfg["targets"], mcfg["r"], mcfg["alpha"], 0.0, cfg["seed"], protect_k=None)
        load_adapter_factors(ref, factors)
        ref.eval()
        logit_ref, mask = _logits(ref, tok, CHECK_TEXTS)
        del ref
        merged32 = _base(cfg, torch.float32, smoke, mcfg["kind"])
        merge_factors_into(merged32, factors, scaling)
        merged32.eval()
        logit_m32, _ = _logits(merged32, tok, CHECK_TEXTS)
        model = merged32.half()
        logit_m16, _ = _logits(model, tok, CHECK_TEXTS)
        info["merge_check"] = {
            "max_abs_diff_fp32_merged_vs_unmerged": float((logit_m32 - logit_ref)[mask].abs().max()),
            "max_abs_diff_fp16_merged_vs_fp32_unmerged": float((logit_m16 - logit_ref)[mask].abs().max()),
            "argmax_agreement_fp16_merged": float(
                (logit_m16.argmax(-1) == logit_ref.argmax(-1))[mask].float().mean()),
            "logit_abs_scale": float(logit_ref[mask].abs().mean()),
        }
        log(f"[{cond}] merge check: {info['merge_check']}")
    model.config.use_cache = True
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    try:
        from transformers import GenerationConfig
        GenerationConfig.from_pretrained(cfg["model"], revision=cfg["model_revision"]).save_pretrained(out)
    except Exception:
        pass
    save_json(info, RESULTS / "task1" / ("smoke" if smoke else "") / "export" / f"{cond}.json")
    log(f"[{cond}] exported to {out}")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    export(a.condition, a.smoke)
