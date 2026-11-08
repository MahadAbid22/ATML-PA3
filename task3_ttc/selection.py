"""Task 3 Steps 1-2: repeated sampling, quality scoring and answer aggregation (offline).

For each question, each of the ten saved orders and each budget N in {1,2,4,8,16,32}, the first N
rollouts of the order are the candidates. Outcomes are averaged over orders within a question, then
over questions. Token cost = sum of retained token counts of the candidates consumed (FirstAns: the
first candidate only). Main accuracy differences get 95% CIs from 1,000 paired question-level
bootstrap resamples (after averaging over orders).

Usage:  python -m task3_ttc.selection [--dataset math|mathvista|all] [--tag _limitN]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.utils import RESULTS, ROOT, load_json, load_yaml, log, save_json
from task3_ttc.core import Prepared, best_pick, rwc, sc_pick
from task3_ttc.data import CFG

BASE_METHODS = ["first_ans", "self_consistency", "best_logp", "best_selfcert", "rwc_selfcert", "shortest"]
VIS_METHODS = ["best_vision", "rwc_vision", "best_visconf", "rwc_visconf", "best_visconf_token", "rwc_visconf_token"]


def methods_for(P: Prepared):
    return BASE_METHODS + (VIS_METHODS if P.has_vision else [])


def select_all(P: Prepared, q: int, prefix: np.ndarray) -> dict:
    """Return {method: (selected position in prefix, selected group, cost)} for one prefix."""
    g = P.grp[q, prefix]
    T = P.T[q, prefix]
    full_cost = int(T.sum())
    out = {}
    N = len(prefix)

    def by_group(gg):  # earliest supporting rollout in order
        return int(np.nonzero(g == gg)[0][0])

    out["first_ans"] = (0, int(g[0]), int(T[0]))
    w = sc_pick(g)
    out["self_consistency"] = (by_group(w), w, full_cost)
    for name, sc in (("best_logp", P.L[q, prefix]), ("best_selfcert", P.K[q, prefix]), ("shortest", -T.astype(float))):
        i = best_pick(sc)
        out[name] = (i, int(g[i]), full_cost)
    w, _, _ = rwc(g, P.K[q, prefix])
    out["rwc_selfcert"] = (by_group(w), w, full_cost)
    if P.has_vision:
        C, Z, Zt = P.visconf(q, prefix) if N > 1 else (np.zeros(1), np.zeros(1), np.zeros(1))
        for nm, sc in (("vision", P.V[q, prefix]), ("visconf", Z), ("visconf_token", Zt)):
            i = best_pick(sc)
            out[f"best_{nm}"] = (i, int(g[i]), full_cost)
            w, _, _ = rwc(g, sc)
            out[f"rwc_{nm}"] = (by_group(w), w, full_cost)
    return out


def run_dataset(name: str, cfg: dict, tag: str = ""):
    P = Prepared(name, tag)
    budgets, n_orders = cfg["budgets"], cfg["n_orders"]
    methods = methods_for(P)
    Q = len(P.qids)
    acc = {(m, N): np.zeros((Q, n_orders)) for m in methods for N in budgets}
    cost = {(m, N): np.zeros((Q, n_orders)) for m in methods for N in budgets}
    oracle = {N: np.zeros((Q, n_orders)) for N in budgets}
    distinct = {N: np.zeros((Q, n_orders)) for N in budgets}
    records = []
    for q in range(Q):
        for o in range(n_orders):
            order = P.orders[q, o]
            for N in budgets:
                prefix = order[:N]
                sel = select_all(P, q, prefix)
                oracle[N][q, o] = P.correct[q, prefix].any()
                distinct[N][q, o] = len(np.unique(P.grp[q, prefix]))
                for m, (pos, gsel, c) in sel.items():
                    ok = bool(P.group_correct[q][gsel])
                    acc[(m, N)][q, o] = ok
                    cost[(m, N)][q, o] = c
                    records.append((P.qids[q], o, N, m, int(prefix[pos]), int(gsel), ok, c))
        if (q + 1) % 100 == 0:
            log(f"[{name}] {q + 1}/{Q} questions")
    df = pd.DataFrame.from_records(records, columns=["qid", "order", "N", "method", "selected_rollout",
                                                     "selected_group", "correct", "token_cost"])
    out_dir = RESULTS / "task3" / ("smoke" if tag else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"selection_records_{name}.parquet", index=False, compression="zstd")

    splits = load_json(ROOT / "data" / "task3" / f"splits_{name}.json") if not tag else \
        {"dev": P.qids[: len(P.qids) // 5], "eval": P.qids[len(P.qids) // 5:]}
    summary = {}
    rng_seed = cfg["seed"]
    for split in ("dev", "eval"):
        idx = np.array([P.qindex[x] for x in splits[split] if x in P.qindex])
        s = {"n_questions": int(len(idx)), "budgets": {}}
        for N in budgets:
            b = {"oracle_coverage": float(oracle[N][idx].mean()), "distinct_groups": float(distinct[N][idx].mean()),
                 "methods": {}}
            for m in methods:
                a = acc[(m, N)][idx]
                b["methods"][m] = {"accuracy": float(a.mean()), "token_cost": float(cost[(m, N)][idx].mean()),
                                   "selection_miss": float(((oracle[N][idx] == 1) & (a == 0)).mean()),
                                   "no_correct_candidate": float((oracle[N][idx] == 0).mean())}
            s["budgets"][str(N)] = b
        # paired bootstrap of main differences (question level, after averaging over orders)
        rng = np.random.default_rng(rng_seed)
        boot_idx = rng.integers(0, len(idx), size=(cfg["bootstrap"], len(idx)))
        contrasts = {}
        for N in (8, 32):
            qa = {m: acc[(m, N)][idx].mean(1) for m in methods}
            pairs = [(m, "self_consistency") for m in methods if m != "self_consistency"] + \
                    [("self_consistency", "first_ans"), ("rwc_selfcert", "best_selfcert"),
                     ("best_selfcert", "best_logp")]
            if P.has_vision:
                pairs += [("rwc_visconf", "best_visconf"), ("rwc_visconf", "rwc_selfcert"), ("rwc_visconf", "rwc_vision"),
                          ("best_visconf", "best_selfcert"), ("best_visconf_token", "best_visconf"),
                          ("rwc_visconf_token", "rwc_visconf")]
            for a_, b_ in pairs:
                d = qa[a_] - qa[b_]
                bs = d[boot_idx].mean(1)
                contrasts[f"N{N}:{a_}-{b_}"] = {"diff": float(d.mean()),
                                                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        s["contrasts"] = contrasts
        summary[split] = s
    # per-question summary (averaged over orders), useful for later analysis
    pq_rows = []
    for q in range(Q):
        for N in budgets:
            row = {"qid": P.qids[q], "N": N, "oracle": oracle[N][q].mean(), "distinct": distinct[N][q].mean(),
                   "pool_frac_correct": P.correct[q].mean()}
            for m in methods:
                row[f"acc_{m}"] = acc[(m, N)][q].mean()
                row[f"cost_{m}"] = cost[(m, N)][q].mean()
            pq_rows.append(row)
    pd.DataFrame(pq_rows).to_csv(out_dir / f"per_question_{name}.csv.gz", index=False)
    save_json(summary, out_dir / f"selection_{name}.json")
    e = summary["eval"]["budgets"]
    log(f"[{name}] eval N=32: oracle {e['32']['oracle_coverage']:.3f} | " +
        " ".join(f"{m} {e['32']['methods'][m]['accuracy']:.3f}" for m in methods))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["math", "mathvista", "all"], default="all")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_yaml(CFG)
    for n in (["math", "mathvista"] if a.dataset == "all" else [a.dataset]):
        run_dataset(n, cfg, a.tag)
