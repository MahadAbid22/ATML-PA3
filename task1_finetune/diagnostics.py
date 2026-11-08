"""Task 1 weight-space and activation diagnostics (FP32).

Probe set: 32 GSM8K test questions + 32 HellaSwag validation contexts sampled with seed 6304
before any model is evaluated (python -m task1_finetune.diagnostics --make-probes).

For down_proj in layers 0, 13, 27 and checkpoints {Full FT mid, five final models}:
  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2   (U_k: top-128 left singular vectors of W0)
  eta   = ||dW||_F / ||W0||_F
Activation ratio on cached starting-model inputs z (last prompt token, benchmark format):
  R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8)
dW = W_trained - W0 for Full FT variants; (alpha/r) B A for LoRA; (alpha/r) B~ A~ for OPLoRA.

Usage:  python -m task1_finetune.diagnostics --make-probes
        python -m task1_finetune.diagnostics --run
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from common.chat import render
from common.lora import top_k_bases
from common.models import load_causal_lm
from common.utils import RESULTS, ROOT, load_json, load_yaml, log, save_json
from task1_finetune.export import condition_source
from task1_finetune.train import CFG, run_dir

PROBE_FILE = ROOT / "data" / "task1" / "probe_ids.json"
CONDS = ["full_ft_mid", "full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"]


def make_probes():
    from datasets import load_dataset
    cfg = load_yaml(CFG)
    dcfg = cfg["diagnostics"]
    rng = np.random.default_rng(cfg["seed"])
    n_gsm = len(load_dataset("openai/gsm8k", "main", split="test"))
    n_hs = len(load_dataset("Rowan/hellaswag", split="validation"))
    gsm = sorted(int(i) for i in rng.choice(n_gsm, dcfg["n_probe_math"], replace=False))
    hs = sorted(int(i) for i in rng.choice(n_hs, dcfg["n_probe_nonmath"], replace=False))
    save_json({"seed": cfg["seed"], "gsm8k_test_indices": gsm, "hellaswag_validation_indices": hs,
               "note": "sampled before inspecting any model results"}, PROBE_FILE)
    log(f"probe set saved: {len(gsm)} GSM8K + {len(hs)} HellaSwag")


def probe_prompts(tok) -> dict:
    """Benchmark prompt formats: gsm8k_cot zero-shot chat prompt; HellaSwag stock context."""
    from datasets import load_dataset
    from lm_eval.tasks.hellaswag.utils import process_docs
    ids = load_json(PROBE_FILE)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = process_docs(load_dataset("Rowan/hellaswag", split="validation"))
    math = [render(tok, [{"role": "user", "content": f"Q: {gsm[i]['question']}\nA:"}])
            for i in ids["gsm8k_test_indices"]]
    nonmath = [hs[i]["query"] for i in ids["hellaswag_validation_indices"]]
    return {"math": math, "nonmath": nonmath}


@torch.no_grad()
def cache_inputs(model, tok, prompts: list[str], layers: list[int]) -> dict:
    store, hooks = {}, []
    for l in layers:
        mod = model.model.layers[l].mlp.down_proj
        hooks.append(mod.register_forward_hook(
            lambda m, inp, out, l=l: store.__setitem__(l, inp[0][0, -1].float().cpu())))
    zs = {l: [] for l in layers}
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
        model(**ids)
        for l in layers:
            zs[l].append(store[l])
    for h in hooks:
        h.remove()
    return {l: torch.stack(v) for l, v in zs.items()}


def delta_w(cond: str, cfg: dict, name: str, W0: torch.Tensor, smoke: bool = False) -> torch.Tensor:
    method, update = condition_source(cond, cfg)
    mcfg = cfg["methods"][method]
    rdir = run_dir(method, smoke)
    if mcfg["kind"] == "full":
        sd = torch.load(rdir / f"ckpt_{update}" / "model.pt", map_location="cpu", mmap=True)
        return sd[name].float() - W0
    fac = torch.load(rdir / "adapter_export.pt", map_location="cpu")
    mod = name.rsplit(".weight", 1)[0]
    A, B = fac[f"{mod}.lora_A"].float(), fac[f"{mod}.lora_B"].float()
    return (mcfg["alpha"] / mcfg["r"]) * (B @ A)


def run(smoke: bool = False):
    from transformers import AutoTokenizer
    cfg = load_yaml(CFG)
    dcfg = cfg["diagnostics"]
    layers, k = dcfg["layers"], dcfg["k"]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    model = load_causal_lm(cfg["model"], torch.float32, cfg["model_revision"], "cuda").eval()
    prompts = probe_prompts(tok)
    Z = {dom: cache_inputs(model, tok, ps, layers) for dom, ps in prompts.items()}
    W0s = {l: model.model.layers[l].mlp.down_proj.weight.detach().float().cpu() for l in layers}
    del model
    torch.cuda.empty_cache()

    ws_rows, act_rows, summ = [], [], []
    for l in layers:
        W0 = W0s[l]
        U_k, _ = top_k_bases(W0.cuda(), k)
        U_k = U_k.cpu()
        name = f"model.layers.{l}.mlp.down_proj.weight"
        for cond in (CONDS if not smoke else ["lora_r4", "oplora_r16"]):
            try:
                dW = delta_w(cond, cfg, name, W0, smoke)
            except FileNotFoundError as e:
                log(f"skip {cond}: {e}")
                continue
            nd = float(dW.norm())
            rho = float((U_k.T @ dW).norm() ** 2 / nd ** 2) if nd > 0 else float("nan")
            eta = nd / float(W0.norm())
            ws_rows.append({"condition": cond, "layer": l, "rho_k": rho, "eta": eta, "dW_fro": nd,
                            "W0_fro": float(W0.norm()), "k": k, "random_baseline_rho": k / W0.shape[0]})
            for dom in ("math", "nonmath"):
                z = Z[dom][l]                                   # [n, d_in]
                num = (z @ dW.T).norm(dim=1)
                den = (z @ W0.T).norm(dim=1) + 1e-8
                R = (num / den).numpy()
                for i, rv in enumerate(R):
                    act_rows.append({"condition": cond, "layer": l, "domain": dom, "probe": i, "R": float(rv)})
                summ.append({"condition": cond, "layer": l, "domain": dom, "R_mean": float(R.mean()),
                             "R_median": float(np.median(R)), "R_std": float(R.std(ddof=1))})
    out = RESULTS / "task1" / ("smoke" if smoke else "") / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("weight_space.csv", ws_rows), ("activation_per_probe.csv", act_rows),
                        ("activation_summary.csv", summ)):
        with open(out / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    log(f"diagnostics written: {len(ws_rows)} weight rows, {len(act_rows)} activation rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-probes", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.make_probes:
        make_probes()
    if a.run:
        run(a.smoke)
