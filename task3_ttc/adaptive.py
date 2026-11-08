"""Task 3 Step 5: adaptive sampling (offline simulation of stopping between completed rollouts).

Rollouts are revealed in the saved orders; stopping is checked after n in {4, 8, 16}, otherwise all 32
are used. Both controllers return the current self-consistency answer.
  agreement:      unique, non-NO_ANSWER SC leader with vote share >= tau
  quality-aware:  agreement AND SC winner == RWC winner AND RWC margin >= delta
                  (self-certainty scores for MATH, VisConf for MathVista; scores, calibration and ranks
                  recomputed on revealed candidates only)
Settings are chosen on the development split only: lowest mean token cost whose accuracy is within
1 pp of fixed-32 SC (always-32 fallback included); ties -> higher accuracy, then smaller tau, delta.
They are frozen before the held-out evaluation.

Usage:  python -m task3_ttc.adaptive [--tag _limitN]
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from common.utils import RESULTS, ROOT, load_json, load_yaml, log, save_json
from task3_ttc.core import Prepared, rwc, sc_state
from task3_ttc.data import CFG


def precompute(P: Prepared, qs, checks):
    """For each question/order/checkpoint n: SC state, RWC winner/margin, cost; plus final SC at 32."""
    n_orders = P.orders.shape[1]
    pts = list(checks) + [32]
    info = {}
    for q in qs:
        for o in range(n_orders):
            order = P.orders[q, o]
            for n in pts:
                pre = order[:n]
                g = P.grp[q, pre]
                w, unique, is_ans, share = sc_state(g, P.group_noans[q])
                if P.has_vision:
                    _, Z, _ = P.visconf(q, pre)
                    score = Z
                else:
                    score = P.K[q, pre]
                rw, margin, _ = rwc(g, score)
                info[(q, o, n)] = (w, unique, is_ans, share, rw, margin, int(P.T[q, pre].sum()),
                                   bool(P.group_correct[q][w]))
    return info


def simulate(info, qs, n_orders, checks, tau, delta=None):
    """Return arrays [Q, orders] of correctness, cost, rollouts used."""
    acc = np.zeros((len(qs), n_orders))
    cost = np.zeros_like(acc)
    used = np.zeros_like(acc)
    for i, q in enumerate(qs):
        for o in range(n_orders):
            stop_n = 32
            for n in checks:
                w, unique, is_ans, share, rw, margin, c, ok = info[(q, o, n)]
                cond = unique and is_ans and share >= tau - 1e-12
                if delta is not None:
                    cond = cond and (rw == w) and (margin >= delta - 1e-12)
                if cond:
                    stop_n = n
                    break
            w, unique, is_ans, share, rw, margin, c, ok = info[(q, o, stop_n)]
            acc[i, o], cost[i, o], used[i, o] = ok, c, stop_n
    return acc, cost, used


def run(name: str, cfg: dict, tag: str = ""):
    P = Prepared(name, tag)
    acfg = cfg["adaptive"]
    checks = acfg["check_points"]
    n_orders = P.orders.shape[1]
    splits = load_json(ROOT / "data" / "task3" / f"splits_{name}.json") if not tag else \
        {"dev": P.qids[: len(P.qids) // 5], "eval": P.qids[len(P.qids) // 5:]}
    qs = {s: [P.qindex[x] for x in splits[s] if x in P.qindex] for s in ("dev", "eval")}
    info = precompute(P, qs["dev"] + qs["eval"], checks)

    def fixed32(s):
        return np.array([[info[(q, o, 32)][7] for o in range(n_orders)] for q in qs[s]])

    out = {"fixed32_sc": {}}
    for s in ("dev", "eval"):
        a = fixed32(s)
        c = np.array([[info[(q, o, 32)][6] for o in range(n_orders)] for q in qs[s]])
        out["fixed32_sc"][s] = {"accuracy": float(a.mean()), "token_cost": float(c.mean())}
    target = out["fixed32_sc"]["dev"]["accuracy"] - acfg["max_drop_pp"] / 100.0

    families = {"agreement": [(t, None) for t in acfg["tau"]],
                "quality_aware": [(t, d) for t, d in itertools.product(acfg["tau"], acfg["delta"])]}
    for fam, grid in families.items():
        rows = [{"tau": None, "delta": None, "fallback_always_32": True,
                 "accuracy": out["fixed32_sc"]["dev"]["accuracy"], "token_cost": out["fixed32_sc"]["dev"]["token_cost"],
                 "mean_rollouts": 32.0}]
        for tau, delta in grid:
            a, c, u = simulate(info, qs["dev"], n_orders, checks, tau, delta)
            rows.append({"tau": tau, "delta": delta, "fallback_always_32": False, "accuracy": float(a.mean()),
                         "token_cost": float(c.mean()), "mean_rollouts": float(u.mean())})
        ok = [r for r in rows if r["accuracy"] >= target - 1e-12]
        ok.sort(key=lambda r: (r["token_cost"], -r["accuracy"], r["tau"] if r["tau"] is not None else 9,
                               r["delta"] if r["delta"] is not None else 9))
        chosen = ok[0]
        # held-out evaluation with the frozen setting
        if chosen["fallback_always_32"]:
            a = fixed32("eval")
            c = np.array([[info[(q, o, 32)][6] for o in range(n_orders)] for q in qs["eval"]])
            u = np.full_like(a, 32.0)
        else:
            a, c, u = simulate(info, qs["eval"], n_orders, checks, chosen["tau"], chosen["delta"])
        f32 = fixed32("eval")
        wrong_early = (a == 0) & (u < 32)
        recover = wrong_early & (f32 == 1)
        stop_dist = {str(n): float((u == n).mean()) for n in checks + [32]}
        # retrospective relation to pool difficulty
        frac = np.array([P.correct[q].mean() for q in qs["eval"]])
        bins = [0, 0.25, 0.5, 0.75, 1.0001]
        by_bin = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (frac >= lo) & (frac < hi)
            if m.any():
                by_bin.append({"pool_frac_correct": [lo, min(hi, 1.0)], "n_questions": int(m.sum()),
                               "mean_rollouts": float(u[m].mean()), "accuracy": float(a[m].mean()),
                               "fixed32_accuracy": float(f32[m].mean())})
        out[fam] = {"dev_grid": rows, "selected": chosen, "dev_target_accuracy": target,
                    "eval": {"accuracy": float(a.mean()), "token_cost": float(c.mean()),
                             "mean_rollouts": float(u.mean()), "stopping_distribution": stop_dist,
                             "incorrect_early_stop_rate": float(wrong_early.mean()),
                             "incorrect_early_stops_fixed32_correct_rate": float(recover.mean()),
                             "n_incorrect_early_stops": int(wrong_early.sum()),
                             "n_recoverable_by_fixed32": int(recover.sum()), "by_pool_difficulty": by_bin}}
        # store early-stop failure examples (first few by question order) for qualitative analysis
        ex = []
        for i, q in enumerate(qs["eval"]):
            for o in range(n_orders):
                if recover[i, o] and len(ex) < 10:
                    ex.append({"qid": P.qids[q], "order": o, "stopped_at": int(u[i, o]),
                               "pool_frac_correct": float(frac[i])})
        out[fam]["eval"]["recoverable_examples"] = ex
        log(f"[{name}] {fam}: selected tau={chosen['tau']} delta={chosen['delta']} -> eval acc "
            f"{a.mean():.3f} cost {c.mean():.0f} (fixed32 {f32.mean():.3f})")
    save_json(out, RESULTS / "task3" / ("smoke" if tag else "") / f"adaptive_{name}.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["math", "mathvista", "all"], default="all")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_yaml(CFG)
    for n in (["math", "mathvista"] if a.dataset == "all" else [a.dataset]):
        run(n, cfg, a.tag)
