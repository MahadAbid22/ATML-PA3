"""Task 3 tables and figures from saved results (held-out evaluation split unless stated).

Usage:  python -m task3_ttc.analyze
"""
from __future__ import annotations

import numpy as np

from common.report import CATEGORICAL, DASHES, MARKERS, NEUTRAL, clean_log_x, fmt, latex_table, savefig, style
from common.utils import RESULTS, load_json, save_json

DS = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
M_LABEL = {"first_ans": "FirstAns", "self_consistency": "Self-consistency", "best_logp": "Best mean log-prob",
           "best_selfcert": "Best self-certainty", "rwc_selfcert": "RWC self-certainty", "shortest": "Shortest (length)",
           "best_vision": "Best vision", "rwc_vision": "RWC vision", "best_visconf": "Best VisConf",
           "rwc_visconf": "RWC VisConf", "best_visconf_token": "Best VisConf (tokenwise)",
           "rwc_visconf_token": "RWC VisConf (tokenwise)"}
ORDER = list(M_LABEL)
COL = {m: CATEGORICAL[i % 8] for i, m in enumerate(["first_ans", "self_consistency", "best_logp", "best_selfcert",
                                                     "rwc_selfcert", "shortest", "best_visconf", "rwc_visconf"])}
COL.update(best_vision=NEUTRAL, rwc_vision="#9c9b96", best_visconf_token="#b7d3f6", rwc_visconf_token="#184f95")


def curves(sel):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    show = ["first_ans", "self_consistency", "best_selfcert", "rwc_selfcert", "best_visconf", "rwc_visconf"]
    for j, ds in enumerate(DS):
        if ds not in sel:
            continue
        b = sel[ds]["eval"]["budgets"]
        Ns = sorted(int(n) for n in b)
        ax, axc = axes[0, j], axes[1, j]
        ax.plot(Ns, [b[str(n)]["oracle_coverage"] for n in Ns], color=INK_ORACLE, lw=1.4, ls="--", marker="o",
                markersize=3, label="Oracle coverage")
        for i, m in enumerate(show):
            if m not in b["1"]["methods"]:
                continue
            ax.plot(Ns, [b[str(n)]["methods"][m]["accuracy"] for n in Ns], color=COL[m], marker=MARKERS[i],
                    markersize=3, ls=DASHES[i % len(DASHES)], label=M_LABEL[m])
            axc.plot([b[str(n)]["methods"][m]["token_cost"] / 1000 for n in Ns],
                     [b[str(n)]["methods"][m]["accuracy"] for n in Ns], color=COL[m], marker=MARKERS[i],
                     markersize=3, ls=DASHES[i % len(DASHES)])
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ns)
        ax.set_xticklabels(Ns)
        ax.set_xlabel("candidates N")
        ax.set_ylabel("accuracy")
        ax.set_title(DS[ds])
        axc.set_xscale("log")
        clean_log_x(axc)
        axc.set_xlabel("mean generated tokens per question (thousands, log)")
        axc.set_ylabel("accuracy")
    h, l = axes[0, 1].get_legend_handles_labels()
    if not h:
        h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "task3_accuracy_coverage")


INK_ORACLE = "#0b0b0b"
T3 = RESULTS / "task3"
T3LIVE = RESULTS / "task3" / "live"


def method_table(sel):
    rows = []
    for ds in DS:
        if ds not in sel:
            continue
        b = sel[ds]["eval"]["budgets"]
        con = sel[ds]["eval"]["contrasts"]
        for m in ORDER:
            if m not in b["32"]["methods"]:
                continue
            r8, r32 = b["8"]["methods"][m], b["32"]["methods"][m]
            c = con.get(f"N32:{m}-self_consistency")
            ci = "--" if c is None else f"{100 * c['diff']:+.1f} [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}]"
            rows.append([DS[ds].split(" ")[0] if m == "first_ans" else "", M_LABEL[m], fmt(r8["accuracy"], 1, True),
                         fmt(r32["accuracy"], 1, True), ci, f"{r32['token_cost'] / 1000:.1f}k"])
        rows.append(["", "Oracle coverage", fmt(b["8"]["oracle_coverage"], 1, True),
                     fmt(b["32"]["oracle_coverage"], 1, True), "--", "--"])
        rows.append("midrule")
    if rows:
        rows.pop()
    latex_table(["Data", "Selector", "Acc@8", "Acc@32", "$\\Delta$ vs SC @32 [95\\% CI]", "Tokens@32"], rows,
                "Task 3 selection methods on the held-out split (800 questions, 10 orders).", "tab:t3-methods",
                "task3_methods", size="\\scriptsize")


def decomposition_figure(sel):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, ds in zip(axes, DS):
        if ds not in sel:
            continue
        b = sel[ds]["eval"]["budgets"]
        Ns = sorted(int(n) for n in b)
        m = "rwc_visconf" if ds == "mathvista" else "rwc_selfcert"
        sc = [b[str(n)]["methods"]["self_consistency"] for n in Ns]
        corr = np.array([x["accuracy"] for x in sc])
        miss = np.array([x["selection_miss"] for x in sc])
        none = np.array([x["no_correct_candidate"] for x in sc])
        x = np.arange(len(Ns))
        ax.bar(x, corr, color=CATEGORICAL[2], edgecolor="white", label="selected correct")
        ax.bar(x, miss, bottom=corr, color=CATEGORICAL[3], edgecolor="white", label="correct available, missed")
        ax.bar(x, none, bottom=corr + miss, color="#d6d5d0", edgecolor="white", label="no correct candidate")
        ax.plot(x, [b[str(n)]["methods"][m]["accuracy"] for n in Ns], color=CATEGORICAL[6], marker="D", markersize=3,
                label=f"{M_LABEL[m]} accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(Ns)
        ax.set_xlabel("candidates N")
        ax.set_title(f"{DS[ds]}: self-consistency")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("fraction of questions")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, "task3_decomposition")


def signals_table():
    rows = []
    for ds in DS:
        p = T3 / f"signals_{ds}.json"
        if not p.exists():
            continue
        s = load_json(p)
        for k, v in s["scores"].items():
            rows.append([DS[ds].split(" ")[0], k.replace("_", "\\_"), fmt(v["top_minus_bottom_quartile_all"], 1, True, True),
                         fmt(v["top_minus_bottom_quartile_mixed"], 1, True, True),
                         "--" if v["spearman_with_correct_mixed"] is None else f"{v['spearman_with_correct_mixed']:+.3f}",
                         "--" if v["spearman_with_length"] is None else f"{v['spearman_with_length']:+.3f}"])
    latex_table(["Data", "Score", "Top$-$bottom Q (all, pp)", "Top$-$bottom Q (mixed, pp)", "$\\rho_s$(correct)",
                 "$\\rho_s$(length)"], rows, "Within-question associations at N=32 (held-out split).",
                "tab:t3-signals", "task3_signals", size="\\scriptsize")


def live_table():
    p = T3LIVE / "summary.json"
    if not p.exists():
        return
    s = load_json(p)
    rows = []
    for k, name in (("single", "Single long (1$\\times$3072)"), ("parallel", "Parallel SC (3$\\times$1024)"),
                    ("refinement", "Refinement (1024 + 2048)")):
        v = s["strategies"][k]
        rows.append([name, fmt(v["accuracy"], 1, True), f"{v['mean_input_tokens']:.0f}", f"{v['mean_output_tokens']:.0f}",
                     f"{v['mean_seconds']:.1f}", fmt(v["truncation_rate_calls"], 1, True),
                     fmt(v["final_extraction_failure_rate"], 1, True)])
    t = s["transitions_initial_to_revised"]
    latex_table(["Strategy", "Acc.", "In tok.", "Out tok.", "Time (s)", "Trunc.", "No extr."], rows,
                f"Live MATH comparison on {s['n_questions']} held-out questions.", "tab:t3-live", "task3_live",
                size="\\scriptsize",
                notes=f"Refinement transitions vs shared initial answer: wrong$\\to$correct {t['wrong_to_correct']}, "
                      f"correct$\\to$wrong {t['correct_to_wrong']}, unchanged correct {t['correct_to_correct']}, "
                      f"unchanged wrong {t['wrong_to_wrong']}.")


def adaptive_outputs(sel):
    import matplotlib.pyplot as plt
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, ds in zip(axes, DS):
        p = T3 / f"adaptive_{ds}.json"
        if not p.exists() or ds not in sel:
            continue
        a = load_json(p)
        b = sel[ds]["eval"]["budgets"]
        Ns = sorted(int(n) for n in b)
        ax.plot([b[str(n)]["methods"]["self_consistency"]["token_cost"] / 1000 for n in Ns],
                [b[str(n)]["methods"]["self_consistency"]["accuracy"] for n in Ns], color=CATEGORICAL[1], marker="o",
                markersize=3, label="fixed-N self-consistency")
        for n in Ns:
            ax.annotate(f"N={n}", (b[str(n)]["methods"]["self_consistency"]["token_cost"] / 1000,
                                   b[str(n)]["methods"]["self_consistency"]["accuracy"]), fontsize=5,
                        xytext=(2, -6), textcoords="offset points")
        ax.margins(y=0.12)  # keeps the lowest N labels clear of the axis
        for i, fam in enumerate(("agreement", "quality_aware")):
            e, s = a[fam]["eval"], a[fam]["selected"]
            ax.scatter([e["token_cost"] / 1000], [e["accuracy"]], color=CATEGORICAL[4 + 2 * i], marker=MARKERS[2 + i],
                       s=30, zorder=3, edgecolor="white", label=f"adaptive: {fam.replace('_', '-')}")
            sd = e["stopping_distribution"]
            rows.append([DS[ds].split(" ")[0] if i == 0 else "", fam.replace("_", "-"),
                         "fallback" if s["fallback_always_32"] else f"{s['tau']}", "--" if s["delta"] is None else f"{s['delta']}",
                         fmt(e["accuracy"], 1, True), f"{e['token_cost'] / 1000:.1f}k", f"{e['mean_rollouts']:.1f}",
                         " / ".join(f"{100 * sd[k]:.0f}" for k in ("4", "8", "16", "32")),
                         fmt(e["incorrect_early_stops_fixed32_correct_rate"], 2, True)])
        f = a["fixed32_sc"]["eval"]
        rows.append(["", "fixed-32 SC", "--", "--", fmt(f["accuracy"], 1, True), f"{f['token_cost'] / 1000:.1f}k", "32.0",
                     "0 / 0 / 0 / 100", "--"])
        rows.append("midrule")
        ax.set_xscale("log")
        clean_log_x(ax)
        ax.set_xlabel("mean generated tokens per question (thousands, log)")
        ax.set_ylabel("accuracy")
        ax.set_title(DS[ds])
        ax.legend(fontsize=6)
    if rows:
        rows.pop()
    fig.tight_layout()
    savefig(fig, "task3_adaptive")
    latex_table(["Data", "Controller", "$\\tau$", "$\\delta$", "Acc.", "Tokens", "Rollouts", "Stop @4/8/16/32 (\\%)",
                 "Recoverable early stops"], rows,
                "Adaptive sampling on the held-out split with development-selected thresholds.", "tab:t3-adaptive",
                "task3_adaptive", size="\\scriptsize")


def main(smoke: bool = False):
    global T3, T3LIVE
    if smoke:
        T3, T3LIVE = RESULTS / "task3" / "smoke", RESULTS / "task3" / "live_smoke"
    style()
    sel = {ds: load_json(T3 / f"selection_{ds}.json") for ds in DS
           if (T3 / f"selection_{ds}.json").exists()}
    curves(sel)
    method_table(sel)
    decomposition_figure(sel)
    signals_table()
    live_table()
    adaptive_outputs(sel)
    save_json({ds: {"eval_N32": {m: v["accuracy"] for m, v in sel[ds]["eval"]["budgets"]["32"]["methods"].items()},
                    "oracle_N32": sel[ds]["eval"]["budgets"]["32"]["oracle_coverage"]} for ds in sel},
              T3 / "summary.json")
    print("task3 analysis written")


if __name__ == "__main__":
    import sys
    main("--smoke" in sys.argv)
