"""Task 2 final evaluation on the two final-test suites under four image conditions.

Conditions: correct image | complementary swap (partner image) | mismatched (fixed seeded mapping to
an image of another pair in the same suite, never either image of the original pair) | no image
(visual tokens removed, masks/positions rebuilt). Questions and original references are fixed.
Greedy decoding, thinking disabled, 16 new tokens, batch 8.

Metrics per suite x condition: accuracy (grounding: normalised exact match, presence / counting;
VQAv2: soft consensus accuracy vs all ten answers, yes/no / number / other), pair accuracy,
identical-answer rate, empty-answer rate, follow-partner rate, mean reference-answer NLL
(termination excluded), generation time per answer. Correct-image advantages over the swap with
95% CIs from 4,000 paired bootstrap resamples of complete pairs.

Usage:  python -m task2_vlm.evaluate [--tokens 196|49] [--smoke]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common.lora import inject_adapters
from common.utils import OUTPUTS, RESULTS, ROOT, gpu_peak_gb, load_yaml, log, save_json, write_jsonl
from task2_vlm.train import CFG, feats_for, load_everything, nll_batch
from task2_vlm.vqa_norm import normalize_answer, soft_accuracy

CONDITIONS = ["correct", "complementary", "mismatched", "none"]


def mismatch_mapping(exs, seed: int, suite: str) -> dict:
    """Seeded derangement over pairs; each question gets an image from a different pair."""
    pairs = sorted({e["pair_id"] for e in exs})
    rng = np.random.default_rng([seed, sum(map(ord, suite))])
    order = rng.permutation(len(pairs))
    by_pair = {}
    for e in exs:
        by_pair.setdefault(e["pair_id"], {})[e["side"]] = e["image"]
    mapping = {}
    pos = {p: i for i, p in enumerate(pairs[k] for k in order)}
    cyc = [pairs[k] for k in order]
    for e in exs:
        own = (by_pair[e["pair_id"]]["a"], by_pair[e["pair_id"]]["b"])
        side = "a" if rng.random() < 0.5 else "b"
        other, img = None, None
        for step in range(1, len(cyc)):  # next pair in the cycle; skip any that shares an image
            cand = cyc[(pos[e["pair_id"]] + step) % len(cyc)]
            for sd in (side, "b" if side == "a" else "a"):
                if by_pair[cand][sd] not in own:
                    other, img, side = cand, by_pair[cand][sd], sd
                    break
            if img is not None:
                break
        assert img is not None and img not in own
        mapping[f"{e['pair_id']}|{e['side']}"] = {"pair": other, "side": side, "image": list(img)}
    return mapping


def score(e, pred) -> float:
    if e["kind"] == "vqa":
        return soft_accuracy(pred, e["answers"])
    return float(normalize_answer(pred) == normalize_answer(e["canonical"]))


def pair_bootstrap(pair_ids, a: np.ndarray, b: np.ndarray | None, n: int, seed: int):
    """Mean of per-pair statistic (or paired difference a-b) with a 95% percentile CI."""
    rng = np.random.default_rng(seed)
    x = a if b is None else a - b
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boots = x[idx].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def run(n_tokens: int = 196, smoke: bool = False):
    cfg = load_yaml(CFG)
    seed, ecfg = cfg["seed"], cfg["eval"]
    tag = f"tokens{n_tokens}" + ("_smoke" if smoke else "")
    rdir = OUTPUTS / "task2" / "runs" / tag
    res = RESULTS / "task2" / tag
    vlm, store, _, evals, _ = load_everything(cfg, n_tokens, smoke)
    w = torch.load(rdir / "stage3_weights.pt", map_location="cpu", weights_only=False)
    vlm.connector.load_state_dict(w["connector"])
    vlm.scale.fill_(w["scale"])
    inject_adapters(vlm.llm, cfg["lora"]["targets"], cfg["lora"]["r"], cfg["lora"]["alpha"], 0.0, seed)
    named = dict(vlm.llm.named_parameters())
    for n, v in w["lora"].items():
        named[n].data.copy_(v)
    vlm.eval()
    torch.cuda.reset_peak_memory_stats()

    report, per_example = {}, []
    for suite in ("grounding_test", "vqa_test"):
        exs = evals[suite]
        mm = mismatch_mapping(exs, seed, suite)
        save_json(mm, res / f"mismatch_mapping_{suite}.json")
        for e in exs:
            e["mismatch_image"] = tuple(mm[f"{e['pair_id']}|{e['side']}"]["image"])
        rows = {c: [] for c in CONDITIONS}
        for cond in CONDITIONS:
            key = {"correct": "image", "complementary": "partner_image", "mismatched": "mismatch_image"}.get(cond)
            nll = nll_batch(vlm, store, exs, cond)
            t0 = time.perf_counter()
            preds, ntoks, ended = [], [], []
            for i in range(0, len(exs), ecfg["batch_size"]):
                chunk = exs[i:i + ecfg["batch_size"]]
                f = feats_for(store, chunk, key) if key else torch.zeros(len(chunk), 196, 768, dtype=torch.float16)
                with torch.autocast("cuda", dtype=torch.float16):
                    t, n, d = vlm.generate(chunk, f, [key is not None] * len(chunk), ecfg["max_new_tokens"])
                preds += t
                ntoks += n
                ended += d
            torch.cuda.synchronize()
            gen_s = time.perf_counter() - t0
            for e, p, n_, d_, l_ in zip(exs, preds, ntoks, ended, nll):
                r = {"suite": suite, "condition": cond, "pair_id": e["pair_id"], "side": e["side"], "kind": e["kind"],
                     "answer_type": e.get("answer_type", e["kind"]), "question": e["question"],
                     "reference": e["answer"], "canonical": e["canonical"], "partner_canonical": e["partner_canonical"],
                     "image": list(e["image"]), "partner_image": list(e["partner_image"]),
                     "prediction": p, "pred_norm": normalize_answer(p), "gen_tokens": n_, "stopped": d_,
                     "score": score(e, p), "ref_nll": float(l_),
                     "follows_partner": float(normalize_answer(p) == normalize_answer(e["partner_canonical"])),
                     "empty": float(normalize_answer(p) == "")}
                rows[cond].append(r)
            per_example += rows[cond]
            rows[cond + "_gen_seconds"] = gen_s

        # ---- aggregate
        out = {}
        pair_ids = sorted({e["pair_id"] for e in exs})
        for cond in CONDITIONS:
            rs = rows[cond]
            by_pair = {}
            for r in rs:
                by_pair.setdefault(r["pair_id"], {})[r["side"]] = r
            pair_ok = np.array([float(all(normalize_answer(by_pair[p][s]["prediction"]) ==
                                          normalize_answer(by_pair[p][s]["canonical"]) for s in "ab")) for p in pair_ids])
            ident = np.array([float(by_pair[p]["a"]["pred_norm"] == by_pair[p]["b"]["pred_norm"]) for p in pair_ids])
            pair_score = np.array([np.mean([by_pair[p][s]["score"] for s in "ab"]) for p in pair_ids])
            types = sorted({r["answer_type"] for r in rs})
            out[cond] = {
                "accuracy": pair_bootstrap(pair_ids, pair_score, None, cfg["eval"]["bootstrap"], seed),
                "accuracy_by_type": {t: float(np.mean([r["score"] for r in rs if r["answer_type"] == t])) for t in types},
                "pair_accuracy": pair_bootstrap(pair_ids, pair_ok, None, cfg["eval"]["bootstrap"], seed),
                "identical_answer_rate": float(ident.mean()),
                "empty_answer_rate": float(np.mean([r["empty"] for r in rs])),
                "follows_partner_rate": float(np.mean([r["follows_partner"] for r in rs])),
                "mean_ref_nll": float(np.mean([r["ref_nll"] for r in rs])),
                "gen_seconds_per_answer": rows[cond + "_gen_seconds"] / len(rs),
                "n_questions": len(rs), "n_pairs": len(pair_ids)}
        # paired advantages of the correct image over the complementary swap (and other controls)
        adv = {}
        for other in ("complementary", "mismatched", "none"):
            def per_pair(cond, field):
                d = {}
                for r in rows[cond]:
                    d.setdefault(r["pair_id"], []).append(r[field])
                return np.array([np.mean(d[p]) for p in pair_ids])
            adv[other] = {
                "accuracy_correct_minus": pair_bootstrap(pair_ids, per_pair("correct", "score"),
                                                         per_pair(other, "score"), cfg["eval"]["bootstrap"], seed),
                "nll_minus_correct": pair_bootstrap(pair_ids, per_pair(other, "ref_nll"),
                                                    per_pair("correct", "ref_nll"), cfg["eval"]["bootstrap"], seed)}
        out["advantages_of_correct_image"] = adv
        report[suite] = out
        log(f"{suite}: " + ", ".join(f"{c} {out[c]['accuracy']['mean']:.3f}" for c in CONDITIONS))
    report["gpu"] = gpu_peak_gb()
    save_json(report, res / "final_eval.json")
    write_jsonl(per_example, res / "final_predictions.jsonl")
    log("task2 final evaluation written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=196, choices=[196, 49])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.tokens, a.smoke)
