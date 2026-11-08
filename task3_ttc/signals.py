"""Task 3 Step 3: what the signals capture (N = 32, all rollouts of each question).

* Within-question association of each score (and length) with correctness: correctness in the top
  quartile (8 highest) minus bottom quartile (8 lowest) of the score within a question, averaged over
  questions (all questions, and those with mixed correctness); within-question Spearman correlations of
  scores with correctness and with length.
* MathVista: per-rollout Pearson correlation between calibrated confidence c_t and visual engagement
  v_t (calibration pooled over all 32 rollouts), summarised across questions; tokenwise fusion
  Z_token = mean_t(c_t v_t) vs trajectory fusion Z = V C, whose difference is Cov_t(c_t, v_t).
* Illustrative trajectories (fixed rule): the first 3 held-out MathVista questions, in split order,
  whose RWC-VisConf and self-consistency answers at N = 32 (order 0) differ in correctness; for each,
  the highest- and lowest-Z rollouts' token series are saved.

Usage:  python -m task3_ttc.signals [--tag _limitN]
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from common.utils import RESULTS, ROOT, load_json, load_yaml, log, save_json
from task3_ttc.core import Prepared, rwc, sc_pick
from task3_ttc.data import CFG


def quartile_gap(score, correct, k=8):
    """Top-k minus bottom-k correctness by score (stable ordering by rollout id for ties)."""
    order = np.argsort(-score, kind="stable")
    return correct[order[:k]].mean() - correct[order[-k:]].mean()


def analyse(name: str, cfg: dict, tag: str = ""):
    P = Prepared(name, tag)
    splits = load_json(ROOT / "data" / "task3" / f"splits_{name}.json") if not tag else {"eval": P.qids}
    idx = [P.qindex[x] for x in splits["eval"] if x in P.qindex]
    allr = np.arange(32)
    scores = {"logp_L": lambda q: P.L[q], "selfcert_K": lambda q: P.K[q], "neg_length": lambda q: -P.T[q].astype(float)}
    vis_cache = {}
    if P.has_vision:
        for q in idx:
            vis_cache[q] = P.visconf(q, allr)
        scores.update(vision_V=lambda q: P.V[q], calibconf_C=lambda q: vis_cache[q][0],
                      visconf_Z=lambda q: vis_cache[q][1], visconf_token_Ztok=lambda q: vis_cache[q][2])
    res = {"n_questions": len(idx), "scores": {}}
    mixed = [q for q in idx if 0 < P.correct[q].mean() < 1]
    res["n_mixed"] = len(mixed)
    for s, f in scores.items():
        gaps_all = [quartile_gap(f(q), P.correct[q].astype(float)) for q in idx]
        gaps_mix = [quartile_gap(f(q), P.correct[q].astype(float)) for q in mixed]
        rho_c = [spearmanr(f(q), P.correct[q]).statistic for q in mixed]
        rho_len = [spearmanr(f(q), P.T[q]).statistic for q in idx if np.std(f(q)) > 0 and np.std(P.T[q]) > 0]
        res["scores"][s] = {"top_minus_bottom_quartile_all": float(np.mean(gaps_all)),
                            "top_minus_bottom_quartile_mixed": float(np.mean(gaps_mix)) if mixed else None,
                            "spearman_with_correct_mixed": float(np.nanmean(rho_c)) if mixed else None,
                            "spearman_with_length": float(np.nanmean(rho_len)) if rho_len else None}
    # length vs correctness directly
    res["length_correct_vs_incorrect_tokens"] = {
        "mean_tokens_correct": float(np.mean([P.T[q][P.correct[q]].mean() for q in mixed])) if mixed else None,
        "mean_tokens_incorrect": float(np.mean([P.T[q][~P.correct[q]].mean() for q in mixed])) if mixed else None}

    if P.has_vision:
        per_q_corr, corr_correct, corr_wrong, cov = [], [], [], []
        for q in idx:
            kls, vs = P.kl_tok[q], P.v_tok[q]
            pooled = np.concatenate(kls)
            q05, q95 = np.percentile(pooled, cfg["visconf_quantiles"])
            d = max(q95 - q05, 1e-8)
            rs = []
            for r in range(32):
                if len(kls[r]) < 3:
                    continue
                c = np.clip((kls[r] - q05) / d, 0, 1)
                if np.std(c) == 0 or np.std(vs[r]) == 0:
                    continue
                rr = float(np.corrcoef(c, vs[r])[0, 1])
                rs.append(rr)
                (corr_correct if P.correct[q, r] else corr_wrong).append(rr)
            if rs:
                per_q_corr.append(np.mean(rs))
            C, Z, Zt = vis_cache[q]
            cov.extend((Zt - Z).tolist())
        pq = np.array(per_q_corr)
        res["confidence_vision_correlation"] = {
            "per_question_mean": float(pq.mean()), "median": float(np.median(pq)),
            "q10": float(np.percentile(pq, 10)), "q90": float(np.percentile(pq, 90)),
            "frac_questions_positive": float((pq > 0).mean()),
            "mean_correct_rollouts": float(np.mean(corr_correct)) if corr_correct else None,
            "mean_incorrect_rollouts": float(np.mean(corr_wrong)) if corr_wrong else None}
        cov = np.array(cov)
        res["tokenwise_minus_trajectory_fusion"] = {"mean": float(cov.mean()), "mean_abs": float(np.abs(cov).mean()),
                                                    "std": float(cov.std()),
                                                    "frac_positive": float((cov > 0).mean())}
        np.save(RESULTS / "task3" / ("smoke" if tag else "") / "conf_vision_corr_per_question.npy", pq)
        # illustrative trajectories (fixed selection rule)
        chosen = []
        for q in idx:
            order = P.orders[q, 0]
            g = P.grp[q, order]
            w_sc = sc_pick(g)
            C, Z, Zt = P.visconf(q, order)
            w_vc, _, _ = rwc(g, Z)
            if P.group_correct[q][w_sc] != P.group_correct[q][w_vc]:
                hi, lo = order[int(np.argmax(Z))], order[int(np.argmin(Z))]
                q05, q95 = np.percentile(np.concatenate([P.kl_tok[q][r] for r in order]), cfg["visconf_quantiles"])
                d = max(q95 - q05, 1e-8)
                traj = {}
                for lab, r in (("highest_Z", hi), ("lowest_Z", lo)):
                    traj[lab] = {"rollout": int(r), "correct": bool(P.correct[q, r]), "tokens": int(P.T[q, r]),
                                 "c": np.clip((P.kl_tok[q][r] - q05) / d, 0, 1).round(4).tolist(),
                                 "v": P.v_tok[q][r].round(4).tolist()}
                chosen.append({"qid": P.qids[q], "sc_correct": bool(P.group_correct[q][w_sc]),
                               "rwc_visconf_correct": bool(P.group_correct[q][w_vc]), "trajectories": traj})
            if len(chosen) == 3:
                break
        save_json(chosen, RESULTS / "task3" / ("smoke" if tag else "") / "illustrative_trajectories.json", indent=None)
    save_json(res, RESULTS / "task3" / ("smoke" if tag else "") / f"signals_{name}.json")
    log(f"[{name}] signals: " + ", ".join(f"{s} gap {v['top_minus_bottom_quartile_mixed']}" for s, v in res["scores"].items()))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["math", "mathvista", "all"], default="all")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_yaml(CFG)
    for n in (["math", "mathvista"] if a.dataset == "all" else [a.dataset]):
        analyse(n, cfg, a.tag)
