"""Task 1 benchmark evaluation with lm-evaluation-harness (zero-shot, FP16, dropout off).

GSM8K: gsm8k_cot, num_fewshot=0, Qwen3 chat template with thinking disabled, greedy
decoding, 1,024 new tokens, stock stop sequences and answer filters. Primary metric:
flexible-extract exact match (strict-match also stored).
HellaSwag / BoolQ / ARC-Easy: stock completion prompts, no chat template, candidate
log-likelihood scoring; acc and acc_norm stored where available.

Usage:  python -m task1_finetune.evaluate --condition baseline [--limit N] [--smoke]
"""
from __future__ import annotations

import argparse
import gzip
import re
import json
import time

import torch

from common.utils import RESULTS, environment_info, gpu_peak_gb, load_yaml, log, save_json
from task1_finetune.export import model_dir
from task1_finetune.train import CFG

TRUNC_SLACK = 4  # a response within 4 tokens of the cap is flagged as truncated
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _num(s: str | None):
    if s is None:
        return None
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def diagnostic_extract(text: str):
    """Diagnostic only (primary metric stays the stock flexible-extract filter): the number in the last
    \\boxed{...} if present, otherwise the last number in the response. Separates answer-format effects
    from correctness."""
    i = text.rfind("\\boxed{")
    if i >= 0:
        m = NUM.findall(text[i:i + 200].split("}")[0])
        if m:
            return _num(m[-1])
    m = NUM.findall(text)
    return _num(m[-1]) if m else None


def _write_gz(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _by_doc(samples):
    out = {}
    for s in samples:
        out.setdefault(s["doc_id"], []).append(s)
    return out


def evaluate(cond: str, limit: int | None = None, smoke: bool = False, batch_gen=None, batch_ll=None):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    cfg = load_yaml(CFG)
    ecfg = cfg["eval"]
    seed = cfg["seed"]
    path = cfg["model"] if cond == "baseline" else str(model_dir(cond, smoke))
    rev = cfg["model_revision"] if cond == "baseline" else None
    out_dir = RESULTS / "task1" / ("smoke" if smoke else "") / "eval" / cond
    tok = AutoTokenizer.from_pretrained(path, revision=rev)
    max_gen = ecfg["gsm8k_max_gen_toks"]

    torch.cuda.reset_peak_memory_stats()
    lm = HFLM(pretrained=path, revision=rev or "main", dtype="float16", device="cuda",
              batch_size=batch_gen or ecfg["batch_size_gen"], enable_thinking=False)
    lm.model.eval()
    common = dict(num_fewshot=0, log_samples=True, limit=limit, random_seed=seed, numpy_random_seed=seed,
                  torch_random_seed=seed, fewshot_random_seed=seed, bootstrap_iters=1000)
    timing = {}

    # ---- GSM8K (generation, chat template, thinking disabled)
    t0 = time.perf_counter()
    r = simple_evaluate(model=lm, tasks=[ecfg["gsm8k_task"]], apply_chat_template=True, fewshot_as_multiturn=False,
                        gen_kwargs={"max_gen_toks": max_gen, "do_sample": False}, **common)
    timing["gsm8k_seconds"] = time.perf_counter() - t0
    task = ecfg["gsm8k_task"]
    res = r["results"][task]
    rows = []
    for doc_id, ss in sorted(_by_doc(r["samples"][task]).items()):
        s0 = ss[0]
        raw = s0["resps"][0][0] if isinstance(s0["resps"][0], list) else s0["resps"][0]
        n_tok = len(tok(raw, add_special_tokens=False)["input_ids"])
        row = {"doc_id": doc_id, "question": s0["doc"]["question"], "target": s0["target"], "response": raw,
               "response_tokens": n_tok, "truncated": n_tok >= max_gen - TRUNC_SLACK}
        for s in ss:
            f = s.get("filter", "none")
            fr = s["filtered_resps"][0]
            row[f"extracted_{f}"] = fr if isinstance(fr, str) else str(fr)
            row[f"correct_{f}"] = float(s.get("exact_match", 0.0))
        d = diagnostic_extract(raw)
        row["extracted_diagnostic"] = d
        row["correct_diagnostic"] = float(d is not None and _num(str(s0["target"])) == d)
        row["unextractable_flexible"] = row.get("extracted_flexible-extract", "[invalid]") == "[invalid]"
        row["unextractable_strict"] = row.get("extracted_strict-match", "[invalid]") == "[invalid]"
        rows.append(row)
    _write_gz(rows, out_dir / "gsm8k_samples.jsonl.gz")
    n = max(len(rows), 1)
    gsm = {
        "flexible_extract_em": res.get("exact_match,flexible-extract"),
        "strict_match_em": res.get("exact_match,strict-match"),
        "flexible_extract_em_stderr": res.get("exact_match_stderr,flexible-extract"),
        "n": len(rows),
        "truncation_rate": sum(r_["truncated"] for r_ in rows) / n,
        "unextractable_rate_flexible": sum(r_["unextractable_flexible"] for r_ in rows) / n,
        "unextractable_rate_strict": sum(r_["unextractable_strict"] for r_ in rows) / n,
        "mean_response_tokens": sum(r_["response_tokens"] for r_ in rows) / n,
        "diagnostic_boxed_or_last_number_acc": sum(r_["correct_diagnostic"] for r_ in rows) / n,
    }

    # ---- likelihood benchmarks (no chat template)
    if batch_ll:
        lm.batch_size_per_gpu = batch_ll
    t0 = time.perf_counter()
    r2 = simple_evaluate(model=lm, tasks=list(ecfg["likelihood_tasks"]), apply_chat_template=False, **common)
    timing["likelihood_seconds"] = time.perf_counter() - t0
    ll = {}
    for t in ecfg["likelihood_tasks"]:
        rr = r2["results"][t]
        ll[t] = {"acc": rr.get("acc,none"), "acc_stderr": rr.get("acc_stderr,none"),
                 "acc_norm": rr.get("acc_norm,none"), "acc_norm_stderr": rr.get("acc_norm_stderr,none"),
                 "n": len(r2["samples"][t])}
        srows = []
        for s in r2["samples"][t]:
            srows.append({"doc_id": s["doc_id"], "target": s["target"],
                          "loglik": [float(x[0]) for x in s["filtered_resps"]],
                          "acc": s.get("acc"), "acc_norm": s.get("acc_norm")})
        _write_gz(srows, out_dir / f"{t}_samples.jsonl.gz")

    agg = {"condition": cond, "model_path": path, "gsm8k": gsm, **ll, "timing": timing, "gpu": gpu_peak_gb(),
           "lm_eval_config": {"gsm8k_task": task, "num_fewshot": 0, "max_gen_toks": max_gen, "greedy": True,
                              "chat_template_gsm8k": True, "enable_thinking": False, "dtype": "float16",
                              "limit": limit},
           "task_versions": {**r.get("versions", {}), **r2.get("versions", {})},
           "environment": environment_info()}
    save_json(agg, out_dir / "aggregate.json")
    log(f"[{cond}] GSM8K flex {gsm['flexible_extract_em']:.4f} | "
        + " | ".join(f"{t} acc {ll[t]['acc']:.4f}" for t in ll))
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-gen", type=int, default=None)
    ap.add_argument("--batch-ll", type=int, default=None)
    a = ap.parse_args()
    evaluate(a.condition, a.limit, a.smoke, a.batch_gen, a.batch_ll)
