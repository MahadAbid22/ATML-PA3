"""Task 1 qualitative generation on first_plot_questions.yaml.

24 prompts (8 questions x plain / JSON / template) x 7 conditions = 168 greedy responses,
thinking disabled, 512-new-token cap. Also writes the fixed 42-response annotation subset
(first and last plain questions, all formats and conditions) as a CSV with automatic flags
(truncation, JSON validity, repetition, maths framing). Judgements are filled in manually.

Usage:  python -m task1_finetune.qualitative --condition baseline   (one condition)
        python -m task1_finetune.qualitative --collect                (assemble files)
"""
from __future__ import annotations

import argparse
import csv
import json
import re

import torch
import yaml

from common.chat import end_of_turn_id, render
from common.models import load_causal_lm
from common.utils import RESULTS, ROOT, load_yaml, log, read_jsonl, write_jsonl
from task1_finetune.export import model_dir
from task1_finetune.train import CFG

FORMATS = ("plain", "json", "template")
MATH_PAT = re.compile(r"\\boxed|\\frac|\\times|\$[^$\n]+\$|\bthe answer is\b|\bstep \d|\bcalculat\w*|\bequation\w*|"
                      r"\bsolv(?:e|ing)\b|\d+(?:\.\d+)?\s*[\+\-\*/×=]\s*\d+", re.I)


def load_prompts(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = yaml.safe_load(f)
    out = []
    for it in items:
        qid = it["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = re.sub(r"_(json|template)$", "", qid)
        out.append({"id": qid, "base": base, "format": fmt, "system": it.get("system"),
                    "question": it["paraphrases"][0]})
    return out


def generate(cond: str, smoke: bool = False):
    from transformers import AutoTokenizer
    cfg = load_yaml(CFG)
    qcfg = cfg["qualitative"]
    prompts = load_prompts(ROOT / qcfg["file"])
    path = cfg["model"] if cond == "baseline" else str(model_dir(cond, smoke))
    rev = cfg["model_revision"] if cond == "baseline" else None
    tok = AutoTokenizer.from_pretrained(path, revision=rev)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(path, torch.float16, rev, "cuda").eval()
    eot = end_of_turn_id(tok)
    texts = []
    for p in prompts:
        msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + \
               [{"role": "user", "content": p["question"]}]
        texts.append(render(tok, msgs))
    cap = qcfg["max_new_tokens"]
    rows = []
    bs = 8
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=cap, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=[eot, tok.eos_token_id])
        new = out[:, enc["input_ids"].shape[1]:]
        for j, seq in enumerate(new):
            ids = seq.tolist()
            ended = any(t in (eot, tok.eos_token_id) for t in ids)
            if ended:
                cut = min(k for k, t in enumerate(ids) if t in (eot, tok.eos_token_id))
                ids = ids[:cut]
            p = prompts[i + j]
            rows.append({"condition": cond, "id": p["id"], "base": p["base"], "format": p["format"],
                         "system": p["system"], "question": p["question"],
                         "response": tok.decode(ids, skip_special_tokens=True),
                         "response_tokens": len(ids), "truncated": not ended})
    write_jsonl(rows, RESULTS / "task1" / ("smoke" if smoke else "") / "qualitative" / "by_condition" / f"{cond}.jsonl")
    log(f"[{cond}] {len(rows)} qualitative responses, truncated {sum(r['truncated'] for r in rows)}")


def _json_valid(text: str) -> tuple[bool, bool]:
    """(strictly valid JSON, valid after stripping a ```json fence)"""
    def ok(s):
        try:
            json.loads(s)
            return True
        except Exception:
            return False
    s = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    return ok(s), ok(fenced)


def _repetition(text: str, n: int = 4) -> float:
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def collect(smoke: bool = False):
    cfg = load_yaml(CFG)
    base = RESULTS / "task1" / ("smoke" if smoke else "") / "qualitative"
    conds = cfg["eval"]["conditions"]
    rows = []
    for c in conds:
        f = base / "by_condition" / f"{c}.jsonl"
        if f.exists():
            rows += read_jsonl(f)
    for r in rows:
        r["json_valid_strict"], r["json_valid_fenced"] = _json_valid(r["response"])
        r["repetition_4gram"] = round(_repetition(r["response"]), 3)
        r["math_markers"] = len(MATH_PAT.findall(r["response"]))
    # grouped by question and format
    grouped = {}
    for r in rows:
        grouped.setdefault(r["base"], {}).setdefault(r["format"], {})[r["condition"]] = {
            k: r[k] for k in ("response", "response_tokens", "truncated", "json_valid_strict",
                              "json_valid_fenced", "repetition_4gram", "math_markers")}
    with open(base / "responses_grouped.json", "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2, ensure_ascii=False)
    write_jsonl(rows, base / "responses_all.jsonl")
    # 42-response annotation subset
    ann_ids = cfg["qualitative"]["annotate_ids"]
    order = {c: i for i, c in enumerate(conds)}
    sub = [r for r in rows if r["base"] in ann_ids]
    sub.sort(key=lambda r: (ann_ids.index(r["base"]), FORMATS.index(r["format"]), order[r["condition"]]))
    cols = ["base", "format", "condition", "response_tokens", "truncated", "json_valid_strict", "json_valid_fenced",
            "repetition_4gram", "math_markers", "format_compliance(0-2)", "coherence_relevance(0-2)",
            "repetition(0-2)", "unnecessary_math_framing(y/n)", "inappropriate_content(y/n)", "excerpt", "notes",
            "response"]
    with open(base / "annotation_subset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sub:
            w.writerow({**r, "excerpt": "", "notes": ""})
    # human-readable side-by-side view of the subset
    with open(base / "annotation_subset.md", "w", encoding="utf-8") as f:
        for b in ann_ids:
            for fmt in FORMATS:
                f.write(f"\n\n## {b} [{fmt}]\n")
                for r in (x for x in sub if x["base"] == b and x["format"] == fmt):
                    flag = " **TRUNCATED**" if r["truncated"] else ""
                    f.write(f"\n### {r['condition']} ({r['response_tokens']} tokens){flag}\n\n```text\n"
                            f"{r['response']}\n```\n")
    log(f"collected {len(rows)} responses; annotation subset {len(sub)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect(a.smoke)
    else:
        generate(a.condition, a.smoke)
