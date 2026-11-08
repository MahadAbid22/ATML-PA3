"""Task 1 tables and figures from saved results (no GPU needed).

Outputs: report/tables/task1_*.tex, report/figures/task1_*.{pdf,png}, results/task1/summary.json
Usage:   python -m task1_finetune.analyze
"""
from __future__ import annotations

import csv

import numpy as np

from common.report import DASHES, MARKERS, NEUTRAL, color_map, fmt, latex_table, savefig, style
from common.utils import RESULTS, load_json, load_yaml, read_jsonl, save_json
from task1_finetune.train import CFG

LABEL = {"baseline": "Baseline", "full_ft_mid": "Full FT (mid)", "full_ft": "Full FT", "l2sp": "Full FT + L2-SP",
         "lora_r4": "LoRA r=4", "lora_r16": "LoRA r=16", "oplora_r16": "OPLoRA r=16"}
METHODS = ["full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"]
BENCH = [("gsm8k", "GSM8K", "flexible_extract_em"), ("hellaswag", "HellaSwag", "acc_norm"),
         ("boolq", "BoolQ", "acc"), ("arc_easy", "ARC-Easy", "acc_norm")]
COL = color_map(METHODS)
COL["full_ft_mid"] = "#9ec5f4"  # lighter step of Full FT's hue: same entity, earlier stage
COL["baseline"] = NEUTRAL
T1 = RESULTS / "task1"  # switched to results/task1/smoke by --smoke


def _score(agg, key, metric):
    return agg["gsm8k"][metric] if key == "gsm8k" else agg[key][metric]


def benchmark_table(cfg):
    conds = cfg["eval"]["conditions"]
    aggs = {c: load_json(T1 / "eval" / c / "aggregate.json")
            for c in conds if (T1 / "eval" / c / "aggregate.json").exists()}
    base = aggs.get("baseline")
    rows, summary = [], {}
    for c in conds:
        if c not in aggs:
            continue
        a, r = aggs[c], [LABEL[c]]
        summary[c] = {}
        for key, name, metric in BENCH:
            s = _score(a, key, metric)
            d = None if base is None or c == "baseline" else s - _score(base, key, metric)
            summary[c][key] = {"score": s, "delta_pp": None if d is None else 100 * d}
            r.append(fmt(s, 1, pct=True) + ("" if d is None else f" ({fmt(d, 1, pct=True, sign=True)})"))
        g = a["gsm8k"]
        r += [fmt(g["strict_match_em"], 1, pct=True), fmt(g["unextractable_rate_flexible"], 1, pct=True),
              fmt(g["truncation_rate"], 1, pct=True)]
        summary[c]["gsm8k_extra"] = {k: g[k] for k in ("strict_match_em", "unextractable_rate_flexible",
                                                       "unextractable_rate_strict", "truncation_rate",
                                                       "mean_response_tokens")}
        summary[c]["raw_acc"] = {k: a[k]["acc"] for k, _, m in BENCH if k != "gsm8k"}
        rows.append(r)
        if c in ("baseline", "full_ft"):
            rows.append("midrule")
    if rows and rows[-1] == "midrule":
        rows.pop()
    latex_table(["Model", "GSM8K", "HellaSwag", "BoolQ", "ARC-E", "GSM8K strict", "No extr.", "Trunc."], rows,
                "Task 1 benchmark scores (\\%) and change from baseline (pp).", "tab:t1-bench", "task1_benchmarks",
                notes="GSM8K: flexible-extract EM; HellaSwag/ARC-E: acc\\_norm; BoolQ: acc. "
                      "No extr./Trunc.: GSM8K unextractable and truncated responses.")
    # raw accuracy alongside normalised (appendix)
    rows2 = []
    for c in conds:
        if c in aggs:
            a = aggs[c]
            rows2.append([LABEL[c], fmt(a["hellaswag"]["acc"], 1, True), fmt(a["hellaswag"]["acc_norm"], 1, True),
                          fmt(a["arc_easy"]["acc"], 1, True), fmt(a["arc_easy"]["acc_norm"], 1, True),
                          fmt(a["boolq"]["acc"], 1, True)])
    latex_table(["Model", "HS acc", "HS acc\\_norm", "ARC-E acc", "ARC-E acc\\_norm", "BoolQ acc"], rows2,
                "Raw and length-normalised accuracy (\\%).", "tab:t1-rawacc", "task1_raw_vs_norm")
    return aggs, summary


def benchmark_figure(aggs):
    import matplotlib.pyplot as plt
    if "baseline" not in aggs:
        return
    conds = [c for c in ["full_ft_mid", "full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"] if c in aggs]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=False)
    for ax, (key, name, metric) in zip(axes, BENCH):
        base = _score(aggs["baseline"], key, metric)
        vals = [100 * (_score(aggs[c], key, metric) - base) for c in conds]
        y = np.arange(len(conds))[::-1]
        ax.barh(y, vals, color=[COL[c] for c in conds], height=0.62, edgecolor="white", linewidth=1.0)
        ax.axvline(0, color=NEUTRAL, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[c] for c in conds] if key == "gsm8k" else [])
        ax.set_title(f"{name} (base {100 * base:.1f}%)")
        ax.set_xlabel("Δ vs baseline (pp)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.28)  # room for the value labels on both sides
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:+.1f}" if v >= 0 else f"{v:+.1f} ", va="center", ha="left" if v >= 0 else "right",
                    fontsize=6)
    fig.tight_layout()
    savefig(fig, "task1_benchmark_deltas")


def loss_figure():
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [2, 1]})
    for i, m in enumerate(METHODS):
        p = T1 / "train_logs" / f"{m}.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        u = np.array([r["update"] for r in rows])
        l = np.array([r["sft_loss"] for r in rows])
        ema, s = [], l[0]
        for v in l:
            s = 0.95 * s + 0.05 * v
            ema.append(s)
        ax.plot(u, l, color=COL[m], alpha=0.18, lw=0.6)
        ax.plot(u, ema, color=COL[m], lw=1.6, ls=DASHES[i], label=LABEL[m])
        if m == "l2sp":
            ax2.plot(u, [r["penalty"] for r in rows], color=COL[m], lw=1.6, label="λ·Ω (L2-SP)")
            ax2.set_ylabel("penalty λΩ")
            ax2.set_xlabel("optimizer update")
            ax2.set_title("L2-SP penalty")
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("SFT loss (nats/token)")
    ax.set_title("Training SFT loss (EMA 0.95; raw faint)")
    ax.legend(ncol=2)
    fig.tight_layout()
    savefig(fig, "task1_loss_curves")


def resources_table(cfg):
    rows, out = [], {}
    for m in METHODS:
        p = T1 / "resources" / f"{m}.json"
        if not p.exists():
            continue
        r = load_json(p)
        cks = r["checkpoints"]
        w = max((c["weights_bytes"] for c in cks), default=0)
        tot = max((c["total_bytes"] for c in cks), default=0)
        ov = r["overhead"]
        extra = []
        if "l2sp_reference_bytes" in ov:
            extra.append(f"ref {ov['l2sp_reference_bytes'] / 2**30:.2f} GB; "
                         f"penalty {ov.get('l2sp_penalty_seconds', 0) / 60:.1f} min")
        if "basis_bytes" in ov and ov.get("basis_bytes"):
            extra.append(f"bases {ov['basis_bytes'] / 2**20:.0f} MB, SVD {ov['basis_seconds']:.1f} s; "
                         f"export {ov.get('export_seconds', 0):.1f} s")
        rows.append([LABEL[m], f"{r['trainable_params'] / 1e6:.2f}M", f"{r['train_seconds'] / 60:.1f}",
                     f"{r['gpu']['peak_allocated_gb']:.1f}", f"{w / 2**20:.0f} MB" if w < 2**30 else f"{w / 2**30:.2f} GB",
                     f"{tot / 2**30:.2f} GB", "; ".join(extra) or "--"])
        out[m] = {"trainable_params": r["trainable_params"], "train_minutes": r["train_seconds"] / 60,
                  "peak_gpu_gb": r["gpu"]["peak_allocated_gb"], "weights_bytes": w, "checkpoint_bytes": tot,
                  "skipped_updates": len(r["skipped_updates"]), "displacement_l2": r.get("displacement_l2"),
                  "overhead": ov}
    latex_table(["Method", "Trainable", "Time (min)", "Peak mem (GB)", "Weights", "Full ckpt", "Method overhead"],
                rows, "Task 1 training resources (one epoch, 1,250 updates).", "tab:t1-resources",
                "task1_resources", align="lrrrrrl", size="\\scriptsize")
    return out


def diagnostics_tables():
    ws_p = T1 / "diagnostics" / "weight_space.csv"
    ac_p = T1 / "diagnostics" / "activation_summary.csv"
    if not ws_p.exists():
        return {}
    ws = list(csv.DictReader(open(ws_p)))
    ac = list(csv.DictReader(open(ac_p)))
    conds = ["full_ft_mid", "full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"]
    layers = sorted({int(r["layer"]) for r in ws})
    rows, out = [], {}
    for c in conds:
        r = [LABEL[c]]
        for l in layers:
            w = next((x for x in ws if x["condition"] == c and int(x["layer"]) == l), None)
            am = next((x for x in ac if x["condition"] == c and int(x["layer"]) == l and x["domain"] == "math"), None)
            an = next((x for x in ac if x["condition"] == c and int(x["layer"]) == l and x["domain"] == "nonmath"), None)
            if w is None:
                r += ["--"] * 4
                continue
            r += [f"{float(w['rho_k']):.3f}", f"{float(w['eta']):.4f}", f"{float(am['R_mean']):.4f}",
                  f"{float(an['R_mean']):.4f}"]
            out.setdefault(c, {})[l] = {"rho_k": float(w["rho_k"]), "eta": float(w["eta"]),
                                        "R_math": float(am["R_mean"]), "R_nonmath": float(an["R_mean"])}
        rows.append(r)
    hdr = ["Model"] + [f"L{l} {h}" for l in layers for h in ("$\\rho_k$", "$\\eta$", "$R_\\text{math}$", "$R_\\text{non}$")]
    rb = float(ws[0]["random_baseline_rho"]) if ws else 0.125
    latex_table(hdr, rows, "down\\_proj diagnostics (k=128) for layers " + ", ".join(map(str, layers)) + ".",
                "tab:t1-diag", "task1_diagnostics", size="\\scriptsize",
                notes=f"A random update would give $\\rho_k \\approx k/d_\\text{{out}} = {rb:.3f}$. "
                      "$R$: mean over 32 probes of $\\|\\Delta W z\\|/\\|W_0 z\\|$.")
    return out


def diagnostics_figure(diag):
    import matplotlib.pyplot as plt
    if not diag:
        return
    conds = [c for c in ["full_ft_mid", "full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"] if c in diag]
    layers = sorted(next(iter(diag.values())).keys())
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    x = np.arange(len(layers))
    wdt = 0.8 / len(conds)
    for i, c in enumerate(conds):
        off = (i - (len(conds) - 1) / 2) * wdt
        axes[0].bar(x + off, [diag[c][l]["rho_k"] for l in layers], wdt * 0.92, color=COL[c], label=LABEL[c])
        axes[1].bar(x + off, [diag[c][l]["eta"] for l in layers], wdt * 0.92, color=COL[c])
        axes[2].scatter([diag[c][l]["R_nonmath"] for l in layers], [diag[c][l]["R_math"] for l in layers],
                        color=COL[c], marker=MARKERS[i], s=22, edgecolor="white", linewidth=0.6)
    axes[0].axhline(0.125, color=NEUTRAL, ls="--", lw=0.8)
    axes[0].text(len(layers) - 0.5, 0.125, " random", va="bottom", ha="right", fontsize=6, color=NEUTRAL)
    for ax, t in zip(axes[:2], ["subspace interference ρ_k", "relative update η (log)"]):
        ax.set_xticks(x)
        ax.set_xticklabels([f"layer {l}" for l in layers])
        ax.set_title(t)
        ax.grid(axis="x", visible=False)
    axes[1].set_yscale("log")
    lim = max(max(diag[c][l]["R_math"], diag[c][l]["R_nonmath"]) for c in conds for l in layers) * 1.1
    axes[2].plot([0, lim], [0, lim], color=NEUTRAL, lw=0.8, ls="--")
    axes[2].set_xlabel("R on non-math probes")
    axes[2].set_ylabel("R on math probes")
    axes[2].set_title("activation response (per layer)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "task1_diagnostics")


def annotation_table():
    """Summarise the hand-scored 42-response subset; skipped until every row has been scored."""
    with open(T1 / "qualitative" / "annotation_subset.csv", encoding="utf-8", newline="") as f:
        ann = list(csv.DictReader(f))
    if not ann or any(r["format_compliance(0-2)"] == "" for r in ann):
        return None
    formats = ("plain", "json", "template")
    mean = lambda rows, col: float(np.mean([int(r[col]) for r in rows]))
    count = lambda rows, col: sum(r[col] == "y" for r in rows)
    table, conds = [], {}
    for c, name in LABEL.items():
        sub = [r for r in ann if r["condition"] == c]
        by_fmt = {f: mean([r for r in sub if r["format"] == f], "format_compliance(0-2)") for f in formats}
        s = {"n": len(sub), "format": mean(sub, "format_compliance(0-2)"), "format_by_type": by_fmt,
             "coherence": mean(sub, "coherence_relevance(0-2)"), "repetition": mean(sub, "repetition(0-2)"),
             "math_framing": count(sub, "unnecessary_math_framing(y/n)"),
             "inappropriate": count(sub, "inappropriate_content(y/n)"),
             "truncated": sum(r["truncated"] == "True" for r in sub)}
        conds[c] = s
        table.append([name, fmt(s["format"], 2), " / ".join(fmt(by_fmt[f], 1) for f in formats), fmt(s["coherence"], 2),
                      fmt(s["repetition"], 2), s["math_framing"], s["inappropriate"], s["truncated"]])
    latex_table(["Model", "Format", "Plain / JSON / Tmpl.", "Coherence", "Repetition", "Math", "Inappr.", "Trunc."], table,
                "Hand annotation of the fixed 42-response subset.", "tab:t1-annotation", "task1_annotation",
                size="\\scriptsize", notes="Means over six responses per model (two questions, three formats). Format and "
                "coherence: 2 is best. Repetition: 0 is best. Math, Inappr. and Trunc. are counts out of six.")
    summary = {"scale": {"format": "2 follows the requested format, 1 partly, 0 no",
                         "coherence": "2 clear and answers the prompt, 1 partly on topic or muddled, 0 incoherent or off topic",
                         "repetition": "0 none, 1 some repeated phrasing, 2 heavy looping"},
               "conditions": conds}
    save_json(summary, T1 / "qualitative" / "annotation_summary.json")
    return summary


def main(smoke: bool = False):
    global T1
    if smoke:
        T1 = RESULTS / "task1" / "smoke"
    style()
    cfg = load_yaml(CFG)
    aggs, bench = benchmark_table(cfg)
    benchmark_figure(aggs)
    loss_figure()
    res = resources_table(cfg)
    diag = diagnostics_tables()
    diagnostics_figure(diag)
    annotation_table()
    save_json({"benchmarks": bench, "resources": res, "diagnostics": diag}, T1 / "summary.json")
    print("task1 analysis written")


if __name__ == "__main__":
    import sys
    main("--smoke" in sys.argv)
