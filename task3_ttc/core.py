"""Core selection machinery shared by the offline experiments.

Tie rules (assignment): score / vote ties are resolved by the earliest supporting rollout in the
current order; RWC uses average ranks for tied scores. At N = 1 every rule returns the only candidate.
VisConf calibration (5th / 95th linear-interpolation percentiles of token KL pooled over the current
candidate set) is recomputed for every prefix; unrevealed rollouts are never used.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from task3_ttc.data import PREP

EPS = 1e-8


class Prepared:
    def __init__(self, name: str, tag: str = ""):
        z = np.load(PREP / f"{name}{tag}.npz", allow_pickle=False)
        self.name = name
        self.qids = [str(x) for x in z["qids"]]
        self.qindex = {q: i for i, q in enumerate(self.qids)}
        for k in ("T", "L", "K", "hit", "grp", "correct", "noans", "orders"):
            setattr(self, k, z[k])
        self.has_vision = "V" in z.files
        if self.has_vision:
            self.V = z["V"]
            kl, v, offs = z["kl_vals"], z["v_vals"], z["tok_offs"]
            Q = len(self.qids)
            self.kl_tok = [[kl[offs[q * 32 + r]: offs[q * 32 + r + 1]] for r in range(32)] for q in range(Q)]
            self.v_tok = [[v[offs[q * 32 + r]: offs[q * 32 + r + 1]] for r in range(32)] for q in range(Q)]
        # group-level correctness / no-answer flags
        self.group_correct, self.group_noans = [], []
        for q in range(len(self.qids)):
            G = int(self.grp[q].max()) + 1
            gc = np.zeros(G, bool)
            gn = np.zeros(G, bool)
            gc[self.grp[q]] = self.correct[q]
            gn[self.grp[q]] = self.noans[q]
            self.group_correct.append(gc)
            self.group_noans.append(gn)

    def visconf(self, q: int, prefix: np.ndarray, lo_hi=(5, 95)):
        """(C, Z, Z_token) for the rollouts in `prefix`, calibrated on the prefix only."""
        kls = [self.kl_tok[q][r] for r in prefix]
        vs = [self.v_tok[q][r] for r in prefix]
        pooled = np.concatenate(kls)
        if pooled.size == 0:
            n = len(prefix)
            return np.zeros(n), np.zeros(n), np.zeros(n)
        q05, q95 = np.percentile(pooled, lo_hi)  # linear interpolation (numpy default)
        d = max(q95 - q05, EPS)
        C, Zt = np.empty(len(prefix)), np.empty(len(prefix))
        for i, (k, v) in enumerate(zip(kls, vs)):
            if k.size == 0:
                C[i], Zt[i] = 0.0, 0.0
                continue
            c = np.clip((k - q05) / d, 0.0, 1.0)
            C[i] = c.mean()
            Zt[i] = (c * v).mean()
        Z = self.V[q, prefix] * C
        return C, Z, Zt


def sc_pick(g: np.ndarray) -> int:
    """Majority vote over answer groups; ties -> group of the earliest supporting rollout."""
    counts = np.bincount(g)
    m = counts.max()
    for gg in g:
        if counts[gg] == m:
            return int(gg)
    raise RuntimeError


def sc_state(g: np.ndarray, noans_groups: np.ndarray):
    """(winner, unique_leader, leader_is_answer, leader_share)."""
    counts = np.bincount(g)
    m = counts.max()
    w = sc_pick(g)
    return w, int((counts == m).sum()) == 1, not bool(noans_groups[w]), m / len(g)


def best_pick(scores: np.ndarray) -> int:
    """Index (position in prefix) of the highest score; ties -> earliest (np.argmax semantics)."""
    s = np.where(np.isfinite(scores), scores, -np.inf)
    return int(np.argmax(s))


def rwc(g: np.ndarray, scores: np.ndarray):
    """Rank-weighted consensus. Returns (winner group, margin, weights per group)."""
    N = len(g)
    if N == 1:
        return int(g[0]), 1.0, None
    s = np.where(np.isfinite(scores), scores, -1e30)
    r = rankdata(-s, method="average")  # rank 1 = best
    w = (N - r) / (N - 1)
    W = np.bincount(g, weights=w)
    m = W.max()
    winner = None
    for gg in g:  # ties (within float tolerance) -> earliest supporting rollout
        if np.isclose(W[gg], m, rtol=0, atol=1e-12):
            winner = int(gg)
            break
    srt = np.sort(W)[::-1]
    second = srt[1] if len(srt) > 1 else 0.0
    total = W.sum()
    margin = (srt[0] - second) / total if total > 0 else 0.0
    return winner, float(margin), W
