"""Task 2 tables and figures from saved results.

Qualitative selection rule (fixed before looking at outputs): for each final-test suite, in sorted
pair-id order, take (i) the first pair answered correctly on both images under the correct
condition, (ii) the first pair whose two correct-image predictions are identical (one image
necessarily wrong), and (iii) the first pair where the complementary swap reproduces the partner's
answer on both sides.

Usage:  python -m task2_vlm.analyze [--tokens 196]
"""
from __future__ import annotations

import argparse

import numpy as np

from common.report import CATEGORICAL, DASHES, NEUTRAL, fmt, latex_table, savefig, style
from common.utils import RESULTS, load_json, read_jsonl, save_json
from task2_vlm.download import image_path

COND_LABEL = {"correct": "Correct", "complementary": "Compl. swap", "mismatched": "Mismatched", "none": "No image"}
COND_COL = {c: CATEGORICAL[i] for i, c in enumerate(COND_LABEL)}
STAGE_NAMES = ["Caption", "Instruct", "VQA"]


def curves(res, tag):
    import matplotlib.pyplot as plt
    tr = read_jsonl(res / "train_log.jsonl")
    dv = read_jsonl(res / "dev_log.jsonl")
    bounds = [2250, 3750, 4500]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    g = np.array([r["global_update"] for r in tr])
    l = np.array([r["loss"] for r in tr])
    ema, s = [], l[0]
    for v in l:
        s = 0.97 * s + 0.03 * v
        ema.append(s)
    axes[0].plot(g, l, color=CATEGORICAL[0], alpha=0.15, lw=0.5)
    axes[0].plot(g, ema, color=CATEGORICAL[0])
    axes[0].set_title("training loss")
    x = [r["global_update"] for r in dv]
    axes[1].plot(x, [r["caption_dev_loss"] for r in dv], color=CATEGORICAL[0], marker="o")
    axes[1].set_title("dev caption loss")
    for k, (suite, name) in enumerate((("grounding_dev", "grounding"), ("vqa_dev", "VQAv2"))):
        ax = axes[2 + k]
        for j, c in enumerate(("correct", "complementary", "none")):
            ax.plot(x, [r[f"{suite}_nll_{c}"] for r in dv], color=COND_COL[c], ls=DASHES[j], marker="o",
                    markersize=3, label=COND_LABEL[c])
        ax.set_title(f"dev answer NLL ({name})")
    axes[3].legend(fontsize=6)
    for ax in axes:
        for b in bounds[:-1]:
            ax.axvline(b, color=NEUTRAL, lw=0.7, ls=":")
        ax.set_xlabel("update")
    fig.tight_layout()
    savefig(fig, f"task2_curves_{tag}")
    # swap penalty
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for j, (suite, name) in enumerate((("grounding_dev", "grounding"), ("vqa_dev", "VQAv2"))):
        ax.plot(x, [r[f"{suite}_swap_penalty"] for r in dv], color=CATEGORICAL[j], ls=DASHES[j], marker="o",
                markersize=3, label=name)
    for b in bounds[:-1]:
        ax.axvline(b, color=NEUTRAL, lw=0.7, ls=":")
    ax.axhline(0, color=NEUTRAL, lw=0.8)
    ax.set_xlabel("update")
    ax.set_ylabel("ΔNLL = NLL(partner) − NLL(correct)")
    ax.set_title("dev image-swap penalty")
    ax.legend()
    fig.tight_layout()
    savefig(fig, f"task2_swap_penalty_{tag}")


def tables(res, tag):
    ev = load_json(res / "final_eval.json")
    rows = []
    for suite, name in (("grounding_test", "Grounding"), ("vqa_test", "VQAv2")):
        for c in COND_LABEL:
            m = ev[suite][c]
            types = m["accuracy_by_type"]
            tstr = " / ".join(f"{100 * types[t]:.0f}" for t in sorted(types))
            a = m["accuracy"]
            rows.append([name if c == "correct" else "", COND_LABEL[c],
                         f"{100 * a['mean']:.1f} [{100 * a['ci95'][0]:.0f}, {100 * a['ci95'][1]:.0f}]", tstr,
                         fmt(m["pair_accuracy"]["mean"], 1, True), fmt(m["identical_answer_rate"], 1, True),
                         fmt(m["follows_partner_rate"], 1, True), fmt(m["empty_answer_rate"], 1, True),
                         f"{m['mean_ref_nll']:.2f}"])
        rows.append("midrule")
    rows.pop()
    types_note = "Grounding types: counting / presence; VQAv2: number / other / yes/no."
    latex_table(["Suite", "Image", "Acc. [95\\% CI]", "By type", "Pair acc.", "Identical", "Follows partner",
                 "Empty", "Ref. NLL"], rows, "Task 2 final-test results under four image conditions.",
                "tab:t2-final", f"task2_final_{tag}", size="\\scriptsize", notes=types_note)
    rows = []
    for suite, name in (("grounding_test", "Grounding"), ("vqa_test", "VQAv2")):
        adv = ev[suite]["advantages_of_correct_image"]
        for o in ("complementary", "mismatched", "none"):
            a, n = adv[o]["accuracy_correct_minus"], adv[o]["nll_minus_correct"]
            rows.append([name if o == "complementary" else "", COND_LABEL[o],
                         f"{100 * a['mean']:+.1f} [{100 * a['ci95'][0]:+.1f}, {100 * a['ci95'][1]:+.1f}]",
                         f"{n['mean']:+.2f} [{n['ci95'][0]:+.2f}, {n['ci95'][1]:+.2f}]"])
    latex_table(["Suite", "Control", "$\\Delta$Acc (pp)", "$\\Delta$NLL"], rows,
                "Correct-image advantage over each control (paired pair-bootstrap, 4,000 resamples).",
                "tab:t2-adv", f"task2_advantages_{tag}")
    return ev


def resources(res, tag):
    tr = load_json(res / "train_summary.json")
    fc = load_json(RESULTS / "task2" / "feature_caching.json")
    ev = load_json(res / "final_eval.json")
    rows = [["Feature caching (SigLIP)", f"{fc['n_images']} images", f"{(fc['encode_seconds']) / 60:.1f}",
             f"{fc['gpu']['peak_allocated_gb']:.1f}", "--"]]
    for name, sec in tr["train_seconds_by_stage"].items():
        tb = tr["trainable_by_stage"][name]
        rows.append([f"Train: {name}", f"{tb['total'] / 1e6:.2f}M trainable", f"{sec / 60:.1f}",
                     f"{tr['gpu']['peak_allocated_gb']:.1f}", f"conn {tb['connector'] / 1e6:.2f}M, LoRA {tb['lora'] / 1e6:.2f}M"])
    g = np.mean([ev[s][c]["gen_seconds_per_answer"] for s in ("grounding_test", "vqa_test") for c in COND_LABEL])
    rows.append(["Generation (test)", "16 tokens, batch 8", f"{1000 * g:.0f} ms/answer", f"{ev['gpu']['peak_allocated_gb']:.1f}", "--"])
    latex_table(["Phase", "Size", "Time (min)", "Peak mem (GB)", "Notes"], rows, "Task 2 compute resources.",
                "tab:t2-res", f"task2_resources_{tag}", align="lllrl", size="\\scriptsize")


def qualitative(res, tag):
    import matplotlib.pyplot as plt
    from PIL import Image
    preds = read_jsonl(res / "final_predictions.jsonl")
    idx = {}
    for r in preds:
        idx.setdefault((r["suite"], r["pair_id"]), {}).setdefault(r["condition"], {})[r["side"]] = r
    chosen = []
    for suite in ("grounding_test", "vqa_test"):
        pids = sorted(p for s, p in idx if s == suite)
        rules = {
            "both correct": lambda d: all(d["correct"][s]["score"] >= 0.99 for s in "ab"),
            "identical answers": lambda d: d["correct"]["a"]["pred_norm"] == d["correct"]["b"]["pred_norm"],
            "swap follows partner": lambda d: all(d["complementary"][s]["follows_partner"] for s in "ab"),
        }
        used = set()
        for rname, fn in rules.items():  # first matching pair in sorted id order not already shown
            hit = next((p for p in pids if p not in used and fn(idx[(suite, p)])), None)
            if hit is not None:
                used.add(hit)
                chosen.append((suite, hit, rname))
    ex_json = []
    fig, axes = plt.subplots(len(chosen), 2, figsize=(5.2, 2.0 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, (suite, pid, rname) in enumerate(chosen):
        d = idx[(suite, pid)]
        ex_json.append({"suite": suite, "pair_id": pid, "rule": rname,
                        "question": d["correct"]["a"]["question"],
                        "sides": {s: {"reference": d["correct"][s]["reference"],
                                      **{c: d[c][s]["prediction"] for c in d}} for s in "ab"}})
        for col, s in enumerate("ab"):
            ax = axes[row, col]
            try:
                sp, iid = d["correct"][s]["image"]
                ax.imshow(Image.open(image_path(int(iid), sp)).convert("RGB"))
            except Exception:
                ax.text(0.5, 0.5, "image not available", ha="center")
            ax.axis("off")
            ax.set_title(f"ref: {d['correct'][s]['reference']} | pred: {d['correct'][s]['prediction']} | "
                         f"swap: {d['complementary'][s]['prediction']} | none: {d['none'][s]['prediction']}",
                         fontsize=5.5)
        axes[row, 0].text(-0.05, 0.5, f"[{rname}]\n{d['correct']['a']['question']}", transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=6, wrap=True)
    fig.tight_layout()
    savefig(fig, f"task2_qualitative_{tag}")
    save_json(ex_json, res / "qualitative_examples.json")


def main(tag="tokens196"):
    style()
    res = RESULTS / "task2" / tag
    curves(res, tag)
    ev = tables(res, tag)
    resources(res, tag)
    qualitative(res, tag)
    save_json({s: {c: ev[s][c]["accuracy"] for c in COND_LABEL} for s in ("grounding_test", "vqa_test")},
              res / "summary.json")
    print("task2 analysis written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=196)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(f"tokens{a.tokens}" + ("_smoke" if a.smoke else ""))
