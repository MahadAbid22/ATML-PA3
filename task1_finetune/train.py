"""Task 1 training: Full FT, Full FT + L2-SP, LoRA (r=4, 16) and OPLoRA (r=16, k=128).

Shared recipe (assignment): one epoch over 40,000 fixed-order examples, 1,250 updates,
micro-batch 2 x accumulation 16, AdamW(0.9, 0.999, 1e-8), weight decay 0.01 on >=2-D
non-bias parameters, cosine schedule with 38 warm-up updates, global-norm clipping 1.0,
gradient checkpointing, FP16 autocast with dynamic loss scaling (unscale before clipping).
Full-FT variants keep FP32 master parameters; adapter variants use a frozen FP16
backbone with FP32 adapters. Optimizer states are FP32.

SFT loss: next-token cross-entropy summed over response + end-of-turn tokens of all 32
examples of an update and divided by their total count (normalised over the effective
batch). L2-SP adds (lambda/2) * sum_j (theta_j - theta0_j)^2 over all unique parameters;
its exact gradient lambda * (theta - theta0) is added to the unscaled gradient before
clipping, which is identical to back-propagating the penalty.

Usage:  python -m task1_finetune.train --method full_ft [--smoke]
"""
from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.lora import count_trainable, inject_adapters, adapter_state_dict, load_adapter_factors
from common.models import load_causal_lm, tiny_causal_lm, unique_named_parameters
from common.utils import (OUTPUTS, RESULTS, ROOT, dir_size_bytes, environment_info, gpu_peak_gb,
                          load_yaml, log, save_json, seed_everything, write_jsonl)
from task1_finetune.data import DATA_DIR, TrainData

CFG = ROOT / "configs" / "task1.yaml"


def run_dir(method: str, smoke: bool) -> Path:
    return OUTPUTS / "task1" / ("smoke" if smoke else "runs") / method


def build_model(cfg: dict, method_cfg: dict, smoke: bool, device="cuda"):
    kind = method_cfg["kind"]
    dtype = torch.float32 if kind == "full" else torch.float16
    if smoke and kind == "full":  # full FT of the real model does not fit an 8 GB laptop GPU
        model = tiny_causal_lm(cfg["model"], dtype, device, seed=cfg["seed"])
    else:
        model = load_causal_lm(cfg["model"], dtype, cfg["model_revision"], device)
    model.config.use_cache = False
    overhead = {}
    if kind in ("lora", "oplora"):
        for p in model.parameters():
            p.requires_grad_(False)
        overhead = inject_adapters(
            model, cfg["targets"], r=method_cfg["r"], alpha=method_cfg["alpha"],
            dropout=method_cfg["dropout"], seed=cfg["seed"],
            protect_k=method_cfg.get("k") if kind == "oplora" else None)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, overhead


def param_groups(model, wd: float):
    decay, no_decay = [], []
    for n, p in unique_named_parameters(model):
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def sft_loss_sum(model, input_ids, attn, label_mask):
    """Sum of token cross-entropies on target positions (logits only where needed)."""
    hidden = model.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    tgt_mask = label_mask[:, 1:]
    h = hidden[:, :-1][tgt_mask]
    targets = input_ids[:, 1:][tgt_mask]
    logits = model.lm_head(h)
    return F.cross_entropy(logits.float(), targets, reduction="sum")


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def save_checkpoint(path: Path, model, kind, opt, sched, scaler, update, examples_seen, extra):
    path.mkdir(parents=True, exist_ok=True)
    if kind == "full":
        sd = {n: p.detach().cpu() for n, p in unique_named_parameters(model)}
        torch.save(sd, path / "model.pt")
    else:
        torch.save(adapter_state_dict(model, projected=False), path / "adapter.pt")
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng_state(), "update": update, "examples_seen": examples_seen, "extra": extra},
               path / "trainer_state.pt")
    return {
        "update": update,
        "weights_bytes": dir_size_bytes(path / ("model.pt" if kind == "full" else "adapter.pt")),
        "trainer_state_bytes": dir_size_bytes(path / "trainer_state.pt"),
        "total_bytes": dir_size_bytes(path),
    }


def latest_checkpoint(rdir: Path):
    cks = sorted(rdir.glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))
    cks = [c for c in cks if (c / "trainer_state.pt").exists()]
    return cks[-1] if cks else None


@torch.no_grad()
def l2sp_penalty_and_grad(params, refs, lam: float, add_grad: bool):
    omega = torch.zeros((), device=params[0].device, dtype=torch.float32)
    for p, r in zip(params, refs):
        d = p.detach() - r
        omega += 0.5 * (d.float() ** 2).sum()
        if add_grad and p.grad is not None:
            p.grad.add_(d, alpha=lam)
    return omega


@torch.no_grad()
def oplora_constraint_check(model) -> dict:
    """Numerical check of U_k^T B~ = 0 and A~ V_k = 0 (relative Frobenius norms)."""
    from common.lora import lora_modules
    worst_b, worst_a = 0.0, 0.0
    for _, m in lora_modules(model):
        A, B = m.effective_factors()
        worst_b = max(worst_b, float((m.U_k.T @ B).norm() / (B.norm() + 1e-12)))
        worst_a = max(worst_a, float((A @ m.V_k).norm() / (A.norm() + 1e-12)))
    return {"max_rel_UkT_Btilde": worst_b, "max_rel_Atilde_Vk": worst_a}


@torch.no_grad()
def displacement(model, ref_cpu: dict) -> float:
    tot = 0.0
    for n, p in unique_named_parameters(model):
        d = p.detach().float() - ref_cpu[n].to(p.device, torch.float32)
        tot += float((d ** 2).sum())
    return tot ** 0.5


def train(method: str, smoke: bool = False):
    cfg = load_yaml(CFG)
    mcfg = cfg["methods"][method]
    tcfg = dict(cfg["train"])
    if smoke:
        tcfg.update(updates=4, grad_accum=2, warmup=1, checkpoint_updates=[2, 4], midpoint_update=2)
    seed_everything(cfg["seed"])
    rdir = run_dir(method, smoke)
    rdir.mkdir(parents=True, exist_ok=True)
    res_dir = RESULTS / "task1" / ("smoke" if smoke else "")
    log_path = res_dir / "train_logs" / f"{method}.jsonl"

    data = TrainData(DATA_DIR / ("train_limit64.npz" if smoke else "train.npz"))
    mb, acc, U = tcfg["micro_batch"], tcfg["grad_accum"], tcfg["updates"]
    per_update = mb * acc
    assert data.n >= per_update * U, f"need {per_update * U} examples, have {data.n}"

    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    torch.cuda.reset_peak_memory_stats()
    t_build = time.perf_counter()
    model, overhead = build_model(cfg, mcfg, smoke)
    torch.cuda.synchronize()
    overhead["model_build_seconds"] = time.perf_counter() - t_build
    kind = mcfg["kind"]
    trainable = [p for _, p in unique_named_parameters(model) if p.requires_grad]
    n_trainable = count_trainable(model)
    log(f"[{method}] trainable parameters: {n_trainable:,}")

    opt = torch.optim.AdamW(param_groups(model, tcfg["weight_decay"]), lr=mcfg["lr"],
                            betas=tuple(tcfg["betas"]), eps=tcfg["eps"], fused=torch.cuda.is_available())
    sched = get_cosine_schedule_with_warmup(opt, tcfg["warmup"], U)
    scaler = torch.amp.GradScaler("cuda")

    lam = float(mcfg.get("l2sp", 0.0))
    refs = None
    if lam > 0:
        # Reference theta0 kept fixed in FP32 on the GPU; storage/transfer overhead recorded.
        names_params = [(n, p) for n, p in unique_named_parameters(model) if p.requires_grad]
        cpu_ref = {n: p.detach().cpu().clone() for n, p in names_params}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        refs = [cpu_ref[n].to("cuda", non_blocking=False) for n, _ in names_params]
        torch.cuda.synchronize()
        overhead["l2sp_reference_transfer_seconds"] = time.perf_counter() - t0
        overhead["l2sp_reference_bytes"] = sum(r.numel() * r.element_size() for r in refs)
        overhead["l2sp_penalty_seconds"] = 0.0
        del cpu_ref
    ref_cpu_disp = None
    if kind == "full":
        ref_cpu_disp = {n: p.detach().cpu().clone() for n, p in unique_named_parameters(model)}

    start_update, ckpt_info, extra = 0, [], {"train_seconds": 0.0, "ckpt_seconds": 0.0, "skipped_updates": []}
    last = latest_checkpoint(rdir)
    if last is not None:
        st = torch.load(last / "trainer_state.pt", map_location="cpu", weights_only=False)
        if kind == "full":
            sd = torch.load(last / "model.pt", map_location="cpu")
            with torch.no_grad():
                for n, p in unique_named_parameters(model):
                    p.copy_(sd[n])
        else:
            load_adapter_factors(model, torch.load(last / "adapter.pt", map_location="cpu"))
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        start_update, extra = st["update"], st["extra"]
        ckpt_info = extra.get("ckpt_info", [])
        log(f"[{method}] resumed from {last.name} (update {start_update})")
    elif log_path.exists():
        log_path.unlink()

    peak_reset_done = False
    for u in range(start_update, U):
        t_u = time.perf_counter()
        base = u * per_update
        idx_all = list(range(base, base + per_update))
        n_tok = int(sum(int(data.resp_len[i]) for i in idx_all))
        lr_used = sched.get_last_lr()[0]
        loss_acc = torch.zeros((), device="cuda")
        for m in range(acc):
            ids, attn, lmask = data.collate(idx_all[m * mb:(m + 1) * mb], pad_id)
            ids, attn, lmask = ids.cuda(non_blocking=True), attn.cuda(non_blocking=True), lmask.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss_sum = sft_loss_sum(model, ids, attn, lmask)
            scaler.scale(loss_sum / n_tok).backward()
            loss_acc += loss_sum.detach()
        scaler.unscale_(opt)
        omega = None
        if lam > 0:
            torch.cuda.synchronize()
            t_pen = time.perf_counter()
            omega = l2sp_penalty_and_grad(trainable, refs, lam, add_grad=True)
            torch.cuda.synchronize()
            extra.setdefault("l2sp_penalty_seconds", 0.0)
            extra["l2sp_penalty_seconds"] += time.perf_counter() - t_pen
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, tcfg["clip"])
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        sched.step()
        opt.zero_grad(set_to_none=True)
        sft = float(loss_acc) / n_tok
        torch.cuda.synchronize()
        dt = time.perf_counter() - t_u
        extra["train_seconds"] += dt
        if skipped:
            extra["skipped_updates"].append(u + 1)
        row = {"update": u + 1, "examples_seen": (u + 1) * per_update, "lr": lr_used, "sft_loss": sft,
               "n_target_tokens": n_tok, "grad_norm": float(gnorm), "loss_scale": scale_before,
               "skipped": bool(skipped), "seconds": dt}
        if omega is not None:
            om = float(omega)
            row.update(omega=om, penalty=lam * om, total_loss=sft + lam * om)
        write_jsonl([row], log_path, mode="a")
        if (u + 1) % 25 == 0 or u == start_update:
            log(f"[{method}] update {u + 1}/{U} loss {sft:.4f} lr {lr_used:.2e} gnorm {float(gnorm):.3f}"
                f"{' SKIPPED' if skipped else ''} {dt:.2f}s/update")
        if (u + 1) in tcfg["checkpoint_updates"]:
            t_c = time.perf_counter()
            info = None
            extra["ckpt_info"] = ckpt_info
            if kind == "full":
                extra.setdefault("displacement", {})[str(u + 1)] = displacement(model, ref_cpu_disp)
            info = save_checkpoint(rdir / f"ckpt_{u + 1}", model, kind, opt, sched, scaler, u + 1,
                                   (u + 1) * per_update, extra)
            ckpt_info.append(info)
            extra["ckpt_info"] = ckpt_info
            # rewrite trainer state so the stored 'extra' includes this checkpoint's info
            st = torch.load(rdir / f"ckpt_{u + 1}" / "trainer_state.pt", map_location="cpu", weights_only=False)
            st["extra"] = extra
            torch.save(st, rdir / f"ckpt_{u + 1}" / "trainer_state.pt")
            extra["ckpt_seconds"] += time.perf_counter() - t_c
            log(f"[{method}] checkpoint {u + 1}: {info['total_bytes'] / 2**30:.2f} GB")

    if kind in ("lora", "oplora"):
        # Export the factors used for inference (projected A~, B~ for OPLoRA), timing the export.
        torch.cuda.synchronize()
        t_e = time.perf_counter()
        fac = adapter_state_dict(model, projected=True)
        torch.save(fac, rdir / "adapter_export.pt")
        overhead["export_seconds"] = time.perf_counter() - t_e
        overhead["export_bytes"] = dir_size_bytes(rdir / "adapter_export.pt")
        if kind == "oplora":
            overhead["constraint_check"] = oplora_constraint_check(model)
    peak = gpu_peak_gb()
    summary = {
        "method": method, "config": mcfg, "train_config": tcfg, "trainable_params": n_trainable,
        "train_seconds": extra["train_seconds"], "checkpoint_seconds": extra["ckpt_seconds"],
        "skipped_updates": extra["skipped_updates"], "checkpoints": ckpt_info,
        "displacement_l2": extra.get("displacement", {}), "overhead": {**overhead, **{
            k: v for k, v in extra.items() if k.startswith("l2sp_")}},
        "gpu": peak, "resumed_from_update": start_update, "environment": environment_info(),
    }
    if lam > 0:
        om = float(l2sp_penalty_and_grad(trainable, refs, lam, add_grad=False))
        summary["l2sp_final_omega"] = om
        summary["l2sp_final_penalty"] = lam * om
    save_json(summary, res_dir / "resources" / f"{method}.json")
    (rdir / "DONE").write_text("ok")
    log(f"[{method}] done: {extra['train_seconds'] / 60:.1f} min, peak {peak['peak_allocated_gb']:.2f} GB")
    return summary


def cleanup(method: str, cfg=None, smoke=False):
    """After export: drop recovery-only checkpoints and optimizer states (sizes already logged)."""
    cfg = cfg or load_yaml(CFG)
    rdir = run_dir(method, smoke)
    mid = cfg["train"]["midpoint_update"] if not smoke else 2
    final = cfg["train"]["updates"] if not smoke else 4
    for ck in rdir.glob("ckpt_*"):
        u = int(ck.name.split("_")[1])
        if u not in (mid, final):
            shutil.rmtree(ck)
        else:
            (ck / "trainer_state.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cleanup", action="store_true", help="after export: drop recovery checkpoints / optimizer states")
    a = ap.parse_args()
    if a.cleanup:
        cleanup(a.method, smoke=a.smoke)
    else:
        train(a.method, a.smoke)
