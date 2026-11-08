"""Task 3 data preparation: join, verify token alignment, group answers, label correctness,
compute per-rollout signals, make the dev / held-out split and the ten rollout orders.

Signals per rollout i with T_i retained tokens (termination excluded in the export):
  L_i = mean_t logp_{i,t};  K_i = mean_t kl_u_p_{i,t} (self-certainty);
  MathVista: v_{i,t} = img / (img + prompt + generated) attention, V_i = mean_t v_{i,t}.
Split: question ids sorted lexicographically, permuted with default_rng(6304); first 200 = dev,
remaining 800 = held-out evaluation. The first 50 MATH evaluation questions form the live subset.
Orders: for every question (sorted id order) ten permutations of the 32 rollout ids drawn from one
default_rng(6304) stream per dataset; all methods and stopping policies reuse them.

Usage:  python -m task3_ttc.data [--limit-questions N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common.utils import OUTPUTS, RESULTS, ROOT, load_yaml, log, save_json
from task3_ttc.answers import NO_ANSWER, group_math, group_mathvista

CFG = ROOT / "configs" / "task3.yaml"
PREP = OUTPUTS / "task3"
KEYS = ["dataset", "sample_id", "rollout_id"]


def _read(path, cols):
    t = pq.read_table(path, columns=KEYS + [c for c in cols if c not in KEYS])
    return t.sort_by([("sample_id", "ascending"), ("rollout_id", "ascending")])


def _list_stats(col):
    """Per-row lengths, per-row means and flattened float32 values of a list<float> column."""
    arr = col.combine_chunks()
    offs = arr.offsets.to_numpy()
    vals = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    lens = np.diff(offs)
    sums = np.add.reduceat(vals, offs[:-1]) if len(vals) else np.zeros(len(lens))
    sums = np.where(lens > 0, sums, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(lens > 0, sums / np.maximum(lens, 1), -np.inf)
    return lens, means, vals, offs


def _group_worker(args):
    kind, extracted, gold_or_meta = args
    if kind == "math":
        return group_math(extracted, gold_or_meta)
    return group_mathvista(extracted, gold_or_meta)


def _pipe_worker(job, conn):
    conn.send(_group_worker(job))
    conn.close()


def _isolated(job, timeout):
    """Run one question's grouping in a fresh process; returns (result or None, exit code)."""
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_pipe_worker, args=(job, child))
    p.start()
    child.close()
    res = None
    try:
        if parent.poll(timeout):
            res = parent.recv()
    except (EOFError, OSError):
        res = None
    p.join(5)
    if p.is_alive():
        p.kill()
        p.join()
    return res, p.exitcode


def _string_group(extracted, gold):
    """Last-resort grouping for a question whose math-verify grouping crashes or hangs:
    whitespace / \\left\\right / spacing-command-insensitive string equality (recorded in the checks)."""
    def norm(s):
        s = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\ |\s+", "", s)
        return s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    labels, first = [], {}
    for a in extracted:
        a = a.strip() if isinstance(a, str) else None
        if not a:
            labels.append(NO_ANSWER)
            continue
        labels.append(first.setdefault(norm(a), a))
    g = norm(str(gold))
    correct = {l: (l != NO_ANSWER and norm(l) == g) for l in set(labels)}
    correct[NO_ANSWER] = False
    return labels, correct


def run_grouping(jobs, timeout=240):
    """Group all questions in a worker pool. A worker that dies (e.g. a SymPy crash on a pathological
    answer) loses its task without raising in Pool.map, so results are collected per task with a
    timeout; any lost task is retried in an isolated process, and only if that also fails does the
    question fall back to string grouping. Returns (results, list of fallback question indices)."""
    if sys.platform == "win32":
        return [_group_worker(j) for j in jobs], []
    ctx = mp.get_context("spawn")  # the parent runs pyarrow / BLAS thread pools: never fork it
    results, lost = [None] * len(jobs), []
    pool = ctx.Pool(min(12, max(1, mp.cpu_count() - 2)), maxtasksperchild=20)  # leave cores for GPU jobs
    pending = {i: pool.apply_async(_group_worker, (j,)) for i, j in enumerate(jobs)}
    last_progress, done = time.time(), 0
    while pending:  # a task still pending after `timeout` s without any completion is treated as lost
        for i in [i for i, a in pending.items() if a.ready()]:
            a = pending.pop(i)
            try:
                results[i] = a.get()
            except Exception as e:  # noqa: BLE001
                lost.append(i)
                log(f"grouping task {i} raised {type(e).__name__}: {e}")
            done += 1
            last_progress = time.time()
            if done % 100 == 0:
                log(f"grouped {done}/{len(jobs)} questions")
        if pending and time.time() - last_progress > timeout:
            log(f"no progress for {timeout}s; {len(pending)} task(s) lost, retrying them in isolation")
            lost.extend(pending)
            break
        time.sleep(0.5)
    pool.terminate()
    pool.join()
    fallback = []
    for i in lost:
        res, code = _isolated(jobs[i], timeout)
        if res is None:
            log(f"question {i}: isolated retry failed (exit code {code}); string-equality fallback")
            res = _string_group(jobs[i][1], jobs[i][2]) if jobs[i][0] == "math" else _group_worker(jobs[i])
            fallback.append(i)
        results[i] = res
    return results, fallback


def mathvista_meta(cfg):
    """Question type, answer type, precision, choices and answer from the released testmini split
    (metadata columns only; images are not read)."""
    from huggingface_hub import hf_hub_download
    m = cfg["mathvista_meta"]
    p = hf_hub_download(m["repo"], m["file"], repo_type="dataset", revision=m["revision"])
    rows = pq.read_table(p, columns=["pid", "question_type", "answer_type", "precision", "choices", "answer",
                                     "unit"]).to_pylist()
    return {str(r["pid"]): r for r in rows}


def prepare(name: str, cfg: dict, limit_q: int | None = None) -> dict:
    dcfg = cfg["datasets"][name]
    folder = ROOT / cfg["data_dir"] / dcfg["folder"]
    gen = _read(folder / "generations.parquet", ["generated_token_ids", "n_generated_text_tokens",
                                                 "hit_max_new_tokens", "gold_answer", "extracted_answer"])
    prob = _read(folder / "probability_metrics.parquet", ["logp", "kl_u_p"])
    for k in KEYS:
        assert pc.all(pc.equal(gen[k], prob[k])).as_py(), f"join mismatch on {k}"
    tok_lens = np.diff(gen["generated_token_ids"].combine_chunks().offsets.to_numpy())
    n_scalar = gen["n_generated_text_tokens"].to_numpy()
    lp_len, L, _, _ = _list_stats(prob["logp"])
    kl_len, K, kl_vals, kl_offs = _list_stats(prob["kl_u_p"])
    checks = {"rows": gen.num_rows, "logp_len_eq_tokens": bool((lp_len == tok_lens).all()),
              "kl_len_eq_tokens": bool((kl_len == tok_lens).all()),
              "scalar_len_eq_tokens": bool((n_scalar == tok_lens).all()),
              "zero_length_rollouts": int((tok_lens == 0).sum())}
    V = None
    if dcfg["kind"] == "mathvista":
        att = _read(folder / "attention_metrics.parquet", ["img_attn_total", "prompt_text_attn_total",
                                                           "generated_text_attn_total", "n_generated_text_tokens"])
        for k in KEYS:
            assert pc.all(pc.equal(gen[k], att[k])).as_py(), f"attention join mismatch on {k}"
        li, _, img, offs = _list_stats(att["img_attn_total"])
        lp_, _, prm, _ = _list_stats(att["prompt_text_attn_total"])
        lg, _, gat, _ = _list_stats(att["generated_text_attn_total"])
        cnt = att["n_generated_text_tokens"].combine_chunks()
        cnt_vals = cnt.flatten().to_numpy()
        expected = np.concatenate([np.arange(n) for n in tok_lens]) if len(tok_lens) else np.array([])
        checks.update(att_len_eq_tokens=bool((li == tok_lens).all() and (lp_ == tok_lens).all() and (lg == tok_lens).all()),
                      att_counts_are_0_to_N_minus_1=bool(np.array_equal(cnt_vals, expected)))
        den = img + prm + gat
        v_vals = np.where(den > 0, img / np.where(den > 0, den, 1), 0.0).astype(np.float32)
        sums = np.add.reduceat(v_vals, offs[:-1]) if len(v_vals) else np.zeros(len(tok_lens))
        V = np.where(tok_lens > 0, sums / np.maximum(tok_lens, 1), -np.inf)
        assert np.array_equal(offs, kl_offs)
    log(f"[{name}] alignment checks: {checks}")
    assert all(v for k, v in checks.items() if isinstance(v, bool)), checks

    sid = gen["sample_id"].to_pylist()
    rid = gen["rollout_id"].to_numpy()
    qids = sorted(set(sid))  # lexicographic
    assert len(sid) == len(qids) * 32 and all((rid.reshape(-1, 32) == np.arange(32)).all(1))
    Q = len(qids)
    gold = gen["gold_answer"].to_pylist()[::32]
    extracted = gen["extracted_answer"].to_pylist()
    hit = gen["hit_max_new_tokens"].to_numpy().reshape(Q, 32)
    T = tok_lens.reshape(Q, 32)

    qsel = range(Q) if limit_q is None else range(min(Q, limit_q))
    if dcfg["kind"] == "math":
        jobs = [("math", extracted[q * 32:(q + 1) * 32], gold[q]) for q in qsel]
    else:
        meta = mathvista_meta(cfg)
        jobs = [("mathvista", extracted[q * 32:(q + 1) * 32], meta[qids[q]]) for q in qsel]
    grouped, fallback = run_grouping(jobs)
    checks["string_fallback_questions"] = [qids[i] for i in fallback]
    nq = len(grouped)
    labels = [g[0] for g in grouped]
    grp = np.zeros((nq, 32), dtype=np.int16)
    correct = np.zeros((nq, 32), dtype=bool)
    noans = np.zeros((nq, 32), dtype=bool)
    group_names = []
    for q, (lab, corr) in enumerate(grouped):
        names = list(dict.fromkeys(lab))  # first-appearance order by rollout id
        group_names.append(names)
        for r, l in enumerate(lab):
            grp[q, r] = names.index(l)
            correct[q, r] = corr[l]
            noans[q, r] = l == NO_ANSWER

    # split and orders
    perm = np.random.default_rng(cfg["seed"]).permutation(Q)
    order_ids = [qids[i] for i in perm]
    dev_ids = order_ids[: cfg["split"]["n_dev"]]
    eval_ids = order_ids[cfg["split"]["n_dev"]:]
    rng = np.random.default_rng(cfg["seed"])
    orders = np.stack([np.stack([rng.permutation(32) for _ in range(cfg["n_orders"])]) for _ in range(Q)])

    out = {"qids": np.array(qids[:nq]), "T": T[:nq], "L": L.reshape(Q, 32)[:nq], "K": K.reshape(Q, 32)[:nq],
           "hit": hit[:nq], "grp": grp, "correct": correct, "noans": noans, "orders": orders[:nq]}
    if V is not None:
        out["V"] = V.reshape(Q, 32)[:nq]
        end = kl_offs[nq * 32]
        out["kl_vals"], out["v_vals"], out["tok_offs"] = kl_vals[:end], v_vals[:end], kl_offs[: nq * 32 + 1]
    PREP.mkdir(parents=True, exist_ok=True)
    tag = "" if limit_q is None else f"_limit{limit_q}"
    np.savez(PREP / f"{name}{tag}.npz", **out)
    save_json({"group_names": group_names, "gold": gold[:nq], "checks": checks}, PREP / f"{name}{tag}_groups.json",
              indent=None)
    if limit_q is None:
        save_json({"dev": dev_ids, "eval": eval_ids,
                   **({"live": eval_ids[: cfg["live"]["n_questions"]]} if name == "math" else {})},
                  ROOT / "data" / "task3" / f"splits_{name}.json")
        np.savez_compressed(ROOT / "data" / "task3" / f"orders_{name}.npz", qids=np.array(qids), orders=orders)
        stats = {"questions": Q, "rollouts": int(Q * 32), "retained_tokens": int(T.sum()),
                 "capped_rollouts": int(hit.sum()), "no_answer_rollouts": int(noans.sum()),
                 "rollout_accuracy": float(correct.mean()), "oracle_at_32": float(correct.any(1).mean()),
                 "mean_distinct_groups_at_32": float(np.mean([len(n) for n in group_names])), "checks": checks}
        save_json(stats, RESULTS / "task3" / f"data_stats_{name}.json")
        log(f"[{name}] {stats}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["math", "mathvista", "all"], default="all")
    ap.add_argument("--limit-questions", type=int, default=None)
    a = ap.parse_args()
    cfg = load_yaml(CFG)
    for n in (["math", "mathvista"] if a.dataset == "all" else [a.dataset]):
        prepare(n, cfg, a.limit_questions)
