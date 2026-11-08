"""Task 2 curriculum training: caption alignment -> visual instruction tuning -> natural VQA.

One model is continued through all three stages (4,500 updates, 72,000 example presentations).
Micro-batch 8 x accumulation 2 (effective 16); AdamW, weight decay 0.01; global-norm clipping 1.0;
per stage 5 % linear warm-up then cosine decay to 10 % of the peak LR; FP16 autocast with dynamic
loss scaling (unscale before clipping); frozen FP16 Qwen backbone, FP32 connector / LoRA and
optimizer states; gradient checkpointing. Loss: mean token NLL within each example (target +
end-of-turn), averaged over the effective batch (example-weighted).

Usage:  python -m task2_vlm.train [--tokens 196|49] [--smoke]
"""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter

import numpy as np
import torch

from common.lora import count_trainable, inject_adapters, lora_modules
from common.models import load_causal_lm
from common.utils import (OUTPUTS, RESULTS, ROOT, environment_info, gpu_peak_gb, load_json, load_yaml, log,
                          save_json, seed_everything, write_jsonl)
from task2_vlm.features import FEAT
from task2_vlm.model import VLM, Connector, FeatureStore, PromptBuilder, pool_tokens

CFG = ROOT / "configs" / "task2.yaml"


# ---------------------------------------------------------------------------- data helpers
def pair_examples(p: dict, split: str) -> list[dict]:
    out = []
    for s, o in (("a", "b"), ("b", "a")):
        e = {"kind": p["kind"], "question": p["question"], "answer": p[s]["answer"], "image": (split, p[s]["image_id"]),
             "partner_image": (split, p[o]["image_id"]), "partner_answer": p[o]["answer"], "pair_id": p["pair_id"],
             "side": s}
        if p["kind"] == "vqa":
            e.update(answers=p[s]["answers"], canonical=p[s]["canonical"], partner_canonical=p[o]["canonical"],
                     answer_type=p["answer_type"])
        elif p["kind"] in ("presence", "counting"):
            e["canonical"] = p[s]["answer"]
            e["partner_canonical"] = p[o]["answer"]
        out.append(e)
    return out


def caption_example(c: dict, split: str) -> dict:
    return {"kind": "caption", "answer": c["answer"], "image": (split, c["image_id"])}


class StageSampler:
    """Seeded sampler. Stage 1: two shuffled passes over the captions. Later stages: each draw picks a
    component by its probability and contributes two examples (a complete pair or two captions)."""

    def __init__(self, stage_idx: int, stage: dict, pools: dict, seed: int):
        self.mix = stage["mix"]
        self.pools = pools
        self.rng = np.random.default_rng([seed, stage_idx])
        self.caption_only = list(self.mix) == ["caption"]
        if self.caption_only:
            n = len(pools["caption"])
            self.order = np.concatenate([self.rng.permutation(n), self.rng.permutation(n)])
        self.pos = 0
        self.counts = Counter()

    def state(self):
        return {"rng": self.rng.bit_generator.state, "pos": self.pos, "counts": dict(self.counts)}

    def load(self, st):
        self.rng.bit_generator.state = st["rng"]
        self.pos = st["pos"]
        self.counts = Counter(st["counts"])

    def next(self, mb: int) -> list[dict]:
        if self.caption_only:
            while self.pos + mb > len(self.order):  # only reachable in tiny smoke tests (real run = 2 passes)
                self.order = np.concatenate([self.order, self.rng.permutation(len(self.pools["caption"]))])
            idx = self.order[self.pos:self.pos + mb]
            self.pos += mb
            self.counts["caption"] += mb
            return [self.pools["caption"][i] for i in idx]
        names = list(self.mix)
        probs = np.array([self.mix[n] for n in names], dtype=float)
        probs /= probs.sum()
        out = []
        for _ in range(mb // 2):
            comp = names[self.rng.choice(len(names), p=probs)]
            self.counts[f"draw_{comp}"] += 1
            pool = self.pools[comp]
            if comp == "caption":
                for _ in range(2):
                    out.append(pool[self.rng.integers(len(pool))])
            else:
                out.extend(pool[self.rng.integers(len(pool))])
            self.pos += 1
        return out


def lr_lambda(total: int, warm_frac: float, final_frac: float):
    w = max(1, int(round(warm_frac * total)))

    def f(t):
        if t < w:
            return (t + 1) / w
        prog = min(1.0, (t - w) / max(1, total - w))
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def param_groups(named, lr, wd):
    decay = [p for n, p in named if p.requires_grad and p.ndim >= 2 and not n.endswith("bias")]
    nod = [p for n, p in named if p.requires_grad and (p.ndim < 2 or n.endswith("bias"))]
    return [{"params": decay, "lr": lr, "weight_decay": wd}, {"params": nod, "lr": lr, "weight_decay": 0.0}]


# ---------------------------------------------------------------------------- setup
def load_everything(cfg, n_tokens, smoke=False):
    from transformers import AutoTokenizer
    ex = load_json(OUTPUTS / "task2" / "examples.json")
    tok = AutoTokenizer.from_pretrained(cfg["llm"], revision=cfg["llm_revision"])
    builder = PromptBuilder(tok, cfg["prompts"], cfg["caps"])
    llm = load_causal_lm(cfg["llm"], torch.float16, cfg["llm_revision"], "cuda")
    for p in llm.parameters():
        p.requires_grad_(False)
    llm.config.use_cache = False
    store = FeatureStore(FEAT / "features.npy", load_json(FEAT / "index.json"))
    pools = {
        "caption": [caption_example(c, "train2014") for c in ex["captions_train"]],
        "presence": [pair_examples(p, "train2014") for p in ex["presence_pool"]],
        "counting": [pair_examples(p, "train2014") for p in ex["counting_pool"]],
        "vqa": [pair_examples(p, "train2014") for p in ex["vqa_train"]],
    }
    evals = {
        "caption_dev": [caption_example(c, "val2014") for c in ex["captions_dev"]],
        "grounding_dev": [e for p in ex["grounding_dev"] for e in pair_examples(p, "val2014")],
        "vqa_dev": [e for p in ex["vqa_dev"] for e in pair_examples(p, "val2014")],
        "grounding_test": [e for p in ex["grounding_test"] for e in pair_examples(p, "val2014")],
        "vqa_test": [e for p in ex["vqa_test"] for e in pair_examples(p, "val2014")],
    }
    if smoke:  # keep only examples whose features were cached in the smoke run
        ok = lambda e: f"{e['image'][0]}/{e['image'][1]}" in store.index and \
            ("partner_image" not in e or f"{e['partner_image'][0]}/{e['partner_image'][1]}" in store.index)  # noqa
        pools = {k: [x for x in v if all(ok(e) for e in (x if isinstance(x, list) else [x]))] for k, v in pools.items()}
        evals = {k: [e for e in v if ok(e)] for k, v in evals.items()}
        evals = {k: v[: (len(v) // 2) * 2] for k, v in evals.items()}
    torch.manual_seed(cfg["seed"])
    conn = Connector().cuda().float()
    vlm = VLM(llm, conn, builder, n_tokens)
    return vlm, store, pools, evals, tok


@torch.no_grad()
def calibrate(vlm: VLM, store, pools, cfg) -> dict:
    """Scale connector outputs so their mean token L2 norm matches the mean norm of text-token
    embeddings in the training data (training data only; fixed constant thereafter)."""
    rng = np.random.default_rng(cfg["seed"])
    tok = vlm.b.tok
    texts = [e["answer"] for e in pools["caption"][:4000]] + \
        [vlm.b.instruction(p[0]) for comp in ("presence", "counting", "vqa") for p in pools[comp][:1000]]
    ids = [i for t in texts for i in tok(t, add_special_tokens=False)["input_ids"]]
    E = vlm.embed.weight.float()
    text_norm = float(E[torch.tensor(ids, device=E.device)].norm(dim=-1).mean())
    caps = pools["caption"]
    pick = rng.choice(len(caps), min(cfg["train"]["calibration_images"], len(caps)), replace=False)
    feats = store.get([caps[i]["image"] for i in pick]).cuda()
    vis_norm = float(torch.cat([vlm.connector(pool_tokens(feats[i:i + 64].float(), vlm.n_tokens))
                                for i in range(0, len(feats), 64)]).norm(dim=-1).mean())
    scale = text_norm / vis_norm
    vlm.scale.fill_(scale)
    return {"rule": "scale = mean L2 norm of training text-token embeddings / mean L2 norm of initial connector "
                    "output tokens on training images", "text_embedding_norm": text_norm,
            "connector_output_norm_init": vis_norm, "scale": scale, "n_text_tokens": len(ids),
            "n_images": int(len(pick))}


def feats_for(store, exs, key="image"):
    return store.get([e[key] for e in exs])


@torch.no_grad()
def nll_batch(vlm, store, exs, condition, bs=8, exclude_eot=True, targets=None):
    """Teacher-forced mean NLL of each example's original reference under an image condition."""
    out = []
    for i in range(0, len(exs), bs):
        chunk = exs[i:i + bs]
        if condition == "correct":
            f, wi = feats_for(store, chunk), [True] * len(chunk)
        elif condition == "complementary":
            f, wi = feats_for(store, chunk, "partner_image"), [True] * len(chunk)
        elif condition == "mismatched":
            f, wi = feats_for(store, chunk, "mismatch_image"), [True] * len(chunk)
        else:  # no image
            f, wi = torch.zeros(len(chunk), 196, 768, dtype=torch.float16), [False] * len(chunk)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(vlm.example_losses(chunk, f, wi, exclude_eot=exclude_eot).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def dev_eval(vlm, store, evals, step) -> dict:
    from task2_vlm.vqa_norm import normalize_answer, soft_accuracy
    vlm.eval()
    r = {"global_update": step}
    cap = nll_batch(vlm, store, evals["caption_dev"], "correct", exclude_eot=False)
    r["caption_dev_loss"] = float(cap.mean())
    for suite in ("grounding_dev", "vqa_dev"):
        exs = evals[suite]
        nll = {c: nll_batch(vlm, store, exs, c) for c in ("correct", "complementary", "none")}
        for c, v in nll.items():
            r[f"{suite}_nll_{c}"] = float(v.mean())
        r[f"{suite}_swap_penalty"] = float((nll["complementary"] - nll["correct"]).mean())
        preds = []
        for i in range(0, len(exs), 8):
            chunk = exs[i:i + 8]
            with torch.autocast("cuda", dtype=torch.float16):
                t, _, _ = vlm.generate(chunk, feats_for(store, chunk), [True] * len(chunk))
            preds += t
        if suite == "vqa_dev":
            r[f"{suite}_acc_correct"] = float(np.mean([soft_accuracy(p, e["answers"]) for p, e in zip(preds, exs)]))
        else:
            r[f"{suite}_acc_correct"] = float(np.mean([normalize_answer(p) == normalize_answer(e["canonical"])
                                                       for p, e in zip(preds, exs)]))
    vlm.train()
    return r


# ---------------------------------------------------------------------------- training
def train(n_tokens: int = 196, smoke: bool = False):
    cfg = load_yaml(CFG)
    tcfg, stages = cfg["train"], [dict(s) for s in cfg["stages"]]
    if smoke:
        for s in stages:
            s["updates"] = 12
        tcfg = dict(tcfg, checkpoint_every=2, dev_every=4)
    seed_everything(cfg["seed"])
    tag = f"tokens{n_tokens}" + ("_smoke" if smoke else "")
    rdir = OUTPUTS / "task2" / "runs" / tag
    rdir.mkdir(parents=True, exist_ok=True)
    res = RESULTS / "task2" / tag
    torch.cuda.reset_peak_memory_stats()
    t_setup = time.perf_counter()
    vlm, store, pools, evals, tok = load_everything(cfg, n_tokens, smoke)
    vlm.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.llm.train()
    mb, acc = tcfg["micro_batch"], tcfg["grad_accum"]
    total_updates = sum(s["updates"] for s in stages)
    bounds = np.cumsum([0] + [s["updates"] for s in stages])
    dev_points = sorted(set(range(0, total_updates + 1, tcfg["dev_every"])) | set(int(b) for b in bounds[1:]))

    # ---- resume or start
    cks = sorted(rdir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    state = torch.load(cks[-1], map_location="cpu", weights_only=False) if cks else None
    if state is None:
        calib = calibrate(vlm, store, pools, cfg)
        save_json(calib, res / "calibration.json")
        g, logs_meta = 0, {"stage_seconds": {}, "trainable_by_stage": {}, "sanity": {}}
        for f in ("train_log.jsonl", "dev_log.jsonl"):
            (res / f).unlink(missing_ok=True)
        write_jsonl([dev_eval(vlm, store, evals, 0)], res / "dev_log.jsonl")
    else:
        vlm.scale.fill_(state["scale"])
        g, logs_meta = state["global_update"], state["meta"]
        vlm.connector.load_state_dict(state["connector"])
        if state.get("lora") is not None:
            inject_adapters(vlm.llm, cfg["lora"]["targets"], cfg["lora"]["r"], cfg["lora"]["alpha"],
                            cfg["lora"]["dropout"], cfg["seed"])
            named = dict(vlm.llm.named_parameters())
            for n, v in state["lora"].items():
                named[n].data.copy_(v)
        log(f"resuming at global update {g}")
    setup_s = time.perf_counter() - t_setup
    # one dynamic loss scaler for the whole run, so its scale carries over between stages
    scaler = torch.amp.GradScaler("cuda")
    if state is not None:
        scaler.load_state_dict(state["scaler"])

    for si, st in enumerate(stages):
        if g >= bounds[si + 1]:
            if si >= 1 and not any(True for _ in lora_modules(vlm.llm)):
                inject_adapters(vlm.llm, cfg["lora"]["targets"], cfg["lora"]["r"], cfg["lora"]["alpha"],
                                cfg["lora"]["dropout"], cfg["seed"])
            continue
        if si >= 1 and not any(True for _ in lora_modules(vlm.llm)):
            inject_adapters(vlm.llm, cfg["lora"]["targets"], cfg["lora"]["r"], cfg["lora"]["alpha"],
                            cfg["lora"]["dropout"], cfg["seed"])
        lora_named = [(n, p) for n, p in vlm.llm.named_parameters() if "lora_" in n]
        groups = param_groups(list(vlm.connector.named_parameters()), st["lr_connector"], tcfg["weight_decay"])
        if st["lr_lora"]:
            groups += param_groups(lora_named, st["lr_lora"], tcfg["weight_decay"])
        else:
            for _, p in lora_named:
                p.requires_grad_(False)
        trainable = [p for gr in groups for p in gr["params"]]
        opt = torch.optim.AdamW(groups, betas=tuple(tcfg["betas"]), fused=True)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(st["updates"], tcfg["warmup_frac"], tcfg["final_lr_frac"]))
        sampler = StageSampler(si, st, pools, cfg["seed"])
        n_tr = sum(p.numel() for p in trainable)
        logs_meta["trainable_by_stage"][st["name"]] = {
            "connector": sum(p.numel() for p in vlm.connector.parameters()),
            "lora": sum(p.numel() for _, p in lora_named) if st["lr_lora"] else 0, "total": n_tr}
        if state is not None and state["stage"] == si:  # mid-stage resume: weights already restored
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            sampler.load(state["sampler"])
            torch.set_rng_state(state["rng"]["torch"])
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
            random.setstate(state["rng"]["python"])
            state = None
        elif state is not None and state["stage"] < si:  # checkpoint was taken at the previous stage's end
            state = None
        log(f"stage {si + 1} {st['name']}: {st['updates']} updates, trainable {n_tr:,}")
        t_stage = time.perf_counter()
        while g < bounds[si + 1]:
            t_u = time.perf_counter()
            loss_tot = 0.0
            for _ in range(acc):
                exs = sampler.next(mb)
                f = feats_for(store, exs)
                with torch.autocast("cuda", dtype=torch.float16):
                    per_ex = vlm.example_losses(exs, f, [True] * len(exs))
                    loss = per_ex.sum() / (mb * acc)
                scaler.scale(loss).backward()
                loss_tot += float(loss.detach())
            scaler.unscale_(opt)
            finite = all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
            if finite and not logs_meta["sanity"].get(st["name"]):
                logs_meta["sanity"][st["name"]] = {
                    "connector_grad_norm": float(sum(p.grad.norm() ** 2 for p in vlm.connector.parameters()
                                                     if p.grad is not None) ** 0.5),
                    "lora_grad_norm": float(sum(p.grad.norm() ** 2 for _, p in lora_named if p.grad is not None) ** 0.5)
                    if st["lr_lora"] else None,
                    "frozen_backbone_has_grad": any(p.grad is not None for n, p in vlm.llm.named_parameters()
                                                    if "lora_" not in n),
                    "backbone_trainable_params": sum(p.numel() for n, p in vlm.llm.named_parameters()
                                                     if p.requires_grad and "lora_" not in n),
                    "feature_shape": list(f.shape)}
                log(f"sanity {st['name']}: {logs_meta['sanity'][st['name']]}")
            gn = torch.nn.utils.clip_grad_norm_(trainable, tcfg["clip"])
            sb = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped = scaler.get_scale() < sb
            lrs = sched.get_last_lr()
            sched.step()
            opt.zero_grad(set_to_none=True)
            g += 1
            write_jsonl([{"global_update": g, "stage": si + 1, "loss": loss_tot, "lr_connector": lrs[0],
                          "lr_lora": lrs[2] if len(lrs) > 2 else None, "grad_norm": float(gn), "skipped": bool(skipped),
                          "seconds": time.perf_counter() - t_u, "mixture": dict(sampler.counts)}],
                        res / "train_log.jsonl", mode="a")
            if g % 50 == 0 or g == bounds[si] + 1:
                log(f"update {g}/{total_updates} stage {si + 1} loss {loss_tot:.4f} gnorm {float(gn):.2f}")
            if g in dev_points:
                write_jsonl([dev_eval(vlm, store, evals, g)], res / "dev_log.jsonl", mode="a")
            if g % tcfg["checkpoint_every"] == 0 or g == bounds[si + 1]:
                logs_meta["stage_seconds"][st["name"]] = logs_meta["stage_seconds"].get(st["name"], 0.0) + \
                    (time.perf_counter() - t_stage)
                t_stage = time.perf_counter()
                torch.save({"global_update": g, "stage": si, "connector": vlm.connector.state_dict(),
                            "lora": {n: p.detach().cpu() for n, p in lora_named} if lora_named else None,
                            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                            "scaler": scaler.state_dict(), "sampler": sampler.state(), "scale": float(vlm.scale),
                            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                                    "python": random.getstate()}, "meta": logs_meta}, rdir / f"ckpt_{g}.pt")
                for old in sorted(rdir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[:-2]:
                    old.unlink()
        torch.save({"connector": vlm.connector.state_dict(), "scale": float(vlm.scale),
                    "lora": {n: p.detach().cpu() for n, p in lora_named} if lora_named else None,
                    "mixture_counts": dict(sampler.counts)}, rdir / f"stage{si + 1}_weights.pt")
        logs_meta.setdefault("mixture_by_stage", {})[st["name"]] = dict(sampler.counts)

    summary = {"n_visual_tokens": n_tokens, "total_updates": total_updates, "example_presentations":
               total_updates * mb * acc, "train_seconds_by_stage": logs_meta["stage_seconds"],
               "train_seconds_total": sum(logs_meta["stage_seconds"].values()), "setup_seconds": setup_s,
               "trainable_by_stage": logs_meta["trainable_by_stage"], "sanity_checks": logs_meta["sanity"],
               "mixture_by_stage": logs_meta.get("mixture_by_stage"), "gpu": gpu_peak_gb(),
               "environment": environment_info()}
    save_json(summary, res / "train_summary.json")
    (rdir / "DONE").write_text("ok")
    log(f"task2 training done: {summary['train_seconds_total'] / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=196, choices=[196, 49])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    train(a.tokens, a.smoke)
