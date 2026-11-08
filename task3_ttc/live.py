"""Task 3 Step 4: parallel sampling vs a single longer response vs sequential refinement (live).

Qwen/Qwen2.5-3B-Instruct at the pinned revision, FP16, no quantisation, batch size 1, temperature 0.7,
top-p 0.8, top-k 20, repetition penalty 1.0, native chat template with the system message
"You are a helpful assistant." and the problem followed by two newlines and the instruction.
Per question (3,072-token output allowance each):
  single     1 x <= 3,072 new tokens
  parallel   3 x <= 1,024, self-consistency over the three extracted answers
  refinement parallel response #1 (shared) + one critique-and-revision call <= 2,048 new tokens
Every call is independently seeded (6304 + 100 * question_index + call_index). Timing excludes model
loading and is measured with CUDA synchronisation. Answers: last \\boxed{} with balanced braces.

Usage:  python -m task3_ttc.live [--limit N] [--model-override ID]   (resumable; one JSON line per call)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common.utils import RESULTS, ROOT, environment_info, gpu_peak_gb, load_json, load_yaml, log, read_jsonl, save_json, write_jsonl
from task3_ttc.answers import NO_ANSWER, group_math, last_boxed
from task3_ttc.data import CFG

CALLS = ["single", "parallel_0", "parallel_1", "parallel_2", "refine"]


def questions(cfg, smoke=False):
    import pyarrow.parquet as pq
    sp = ROOT / "data" / "task3" / "splits_math.json"
    ids = load_json(sp)["live"] if not smoke else None
    t = pq.read_table(ROOT / cfg["data_dir"] / cfg["datasets"]["math"]["folder"] / "generations.parquet",
                      columns=["sample_id", "rollout_id", "question_text", "gold_answer"]).to_pylist()
    meta = {r["sample_id"]: r for r in t if r["rollout_id"] == 0}
    if ids is None:  # smoke test: any questions
        ids = sorted(meta)[:5]
    return [(i, q, meta[q]["question_text"], meta[q]["gold_answer"]) for i, q in enumerate(ids)]


def generate(model, tok, messages, cap, seed, lcfg):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cap, do_sample=True, temperature=lcfg["temperature"],
                             top_p=lcfg["top_p"], top_k=lcfg["top_k"], repetition_penalty=lcfg["repetition_penalty"],
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    ended = any(t in stops for t in new)
    if ended:
        new = new[: min(k for k, t in enumerate(new) if t in stops)]
    resp = tok.decode(new, skip_special_tokens=True)
    return {"prompt": text, "input_tokens": int(enc["input_ids"].shape[1]), "output_tokens": len(new),
            "truncated": (not ended) and len(new) >= cap, "seconds": dt, "response": resp,
            "extracted": last_boxed(resp)}


def run(limit=None, model_override=None):
    from transformers import AutoTokenizer
    cfg = load_yaml(CFG)
    lcfg = dict(cfg["live"])
    if model_override:  # smoke test: shorter caps
        for k in ("single_cap", "parallel_cap", "refine_cap"):
            lcfg[k] //= 8
    out_path = RESULTS / "task3" / ("live_smoke" if model_override else "live") / "calls.jsonl"
    done = {(r["qid"], r["call"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    mid, rev = (model_override, None) if model_override else (lcfg["model"], lcfg["revision"])
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    t_load = time.perf_counter()
    from common.models import load_causal_lm
    model = load_causal_lm(mid, torch.float16, rev, "cuda").eval()
    load_s = time.perf_counter() - t_load
    torch.cuda.reset_peak_memory_stats()
    qs = questions(cfg, smoke=bool(model_override))
    qs = qs[:limit] if limit else qs
    for qi, qid, problem, gold in qs:
        user = problem + "\n\n" + lcfg["instruction"]
        base = [{"role": "system", "content": lcfg["system"]}, {"role": "user", "content": user}]
        rows = {}
        for k, call in enumerate(CALLS):
            if (qid, call) in done:
                continue
            seed = cfg["seed"] + 100 * qi + k
            if call == "single":
                r = generate(model, tok, base, lcfg["single_cap"], seed, lcfg)
            elif call.startswith("parallel"):
                r = generate(model, tok, base, lcfg["parallel_cap"], seed, lcfg)
            else:
                first = rows.get("parallel_0") or next(x for x in read_jsonl(out_path)
                                                        if x["qid"] == qid and x["call"] == "parallel_0")
                msgs = base + [{"role": "assistant", "content": first["response"]},
                               {"role": "user", "content": lcfg["review"]}]
                r = generate(model, tok, msgs, lcfg["refine_cap"], seed, lcfg)
            r.update(qid=qid, q_index=qi, call=call, seed=seed, gold=gold)
            rows[call] = r
            write_jsonl([r], out_path, mode="a")
        log(f"live {qi + 1}/{len(qs)} {qid}")
    save_json({"model": mid, "revision": rev, "dtype": "float16", "batch_size": 1, "model_load_seconds": load_s,
               "gpu": gpu_peak_gb(), "environment": environment_info()}, out_path.parent / "run_info.json")
    summarize(out_path)


def summarize(path):
    rows = read_jsonl(path)
    by_q = {}
    for r in rows:
        by_q.setdefault(r["qid"], {})[r["call"]] = r
    per_q, strat = [], {s: {"correct": [], "in": [], "out": [], "sec": [], "trunc": [], "extract_fail": []}
                        for s in ("single", "parallel", "refinement")}
    trans = {"wrong_to_correct": 0, "correct_to_wrong": 0, "correct_to_correct": 0, "wrong_to_wrong": 0}
    for qid, d in by_q.items():
        if not all(c in d for c in CALLS):
            continue
        gold = d["single"]["gold"]
        ans = {c: d[c]["extracted"] for c in CALLS}
        labels, corr = group_math([ans[c] for c in CALLS], gold)
        lab = dict(zip(CALLS, labels))
        ok = {c: bool(corr[lab[c]]) for c in CALLS}
        # self-consistency over the three parallel answers (ties -> earliest)
        par = [lab[f"parallel_{k}"] for k in range(3)]
        counts = {l: par.count(l) for l in par}
        m = max(counts.values())
        sc = next(l for l in par if counts[l] == m)
        res = {"single": ok["single"], "parallel": bool(corr[sc]), "refinement": ok["refine"]}
        calls = {"single": ["single"], "parallel": ["parallel_0", "parallel_1", "parallel_2"],
                 "refinement": ["parallel_0", "refine"]}
        for s, cs in calls.items():
            strat[s]["correct"].append(res[s])
            strat[s]["in"].append(sum(d[c]["input_tokens"] for c in cs))
            strat[s]["out"].append(sum(d[c]["output_tokens"] for c in cs))
            strat[s]["sec"].append(sum(d[c]["seconds"] for c in cs))
            strat[s]["trunc"].append(np.mean([d[c]["truncated"] for c in cs]))
            final_call = {"single": "single", "parallel": None, "refinement": "refine"}[s]
            strat[s]["extract_fail"].append(
                float(sc == NO_ANSWER) if final_call is None else float(ans[final_call] is None))
        init, rev = ok["parallel_0"], ok["refine"]
        trans[("correct" if init else "wrong") + "_to_" + ("correct" if rev else "wrong")] += 1
        per_q.append({"qid": qid, **{f"{s}_correct": res[s] for s in res}, "initial_correct": init,
                      "answers": ans, "gold": gold, "revised_changed_answer": lab["refine"] != lab["parallel_0"]})
    summary = {"n_questions": len(per_q), "transitions_initial_to_revised": trans, "strategies": {}}
    for s, v in strat.items():
        if not v["correct"]:
            continue
        summary["strategies"][s] = {"accuracy": float(np.mean(v["correct"])),
                                    "mean_input_tokens": float(np.mean(v["in"])),
                                    "mean_output_tokens": float(np.mean(v["out"])),
                                    "mean_total_tokens": float(np.mean(v["in"]) + np.mean(v["out"])),
                                    "mean_seconds": float(np.mean(v["sec"])),
                                    "truncation_rate_calls": float(np.mean(v["trunc"])),
                                    "final_extraction_failure_rate": float(np.mean(v["extract_fail"]))}
    save_json(summary, path.parent / "summary.json")
    write_jsonl(per_q, path.parent / "per_question.jsonl")
    log(f"live summary: {summary}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-override", default=None, help="smoke test with a small model")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if a.summarize_only:
        summarize(RESULTS / "task3" / "live" / "calls.jsonl")
    else:
        run(a.limit, a.model_override)
