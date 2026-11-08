"""MetaMathQA selection and tokenisation for Task 1.

Rules (assignment): first 40,000 eligible examples in source order; exclude empty
questions/responses, formatted length > 2,048 tokens, and examples whose query or
original_question matches a GSM8K test question after text normalisation. No answer
truncation, no shuffling. Each example = user question + supplied assistant response,
formatted with the Qwen3 chat template (thinking disabled). Loss is computed on the
response tokens and the end-of-turn token only.

Usage:  python -m task1_finetune.data [--limit N]
"""
from __future__ import annotations

import argparse
import re
import unicodedata

import numpy as np

from common.chat import encode, end_of_turn_id, render
from common.utils import OUTPUTS, ROOT, RESULTS, load_yaml, log, save_json

CFG = ROOT / "configs" / "task1.yaml"
DATA_DIR = OUTPUTS / "task1" / "data"


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def build(limit: int | None = None) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    cfg = load_yaml(CFG)
    n_target = limit or cfg["data"]["n_examples"]
    max_len = cfg["data"]["max_len"]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    eot = end_of_turn_id(tok)

    gsm = load_dataset(cfg["data"]["gsm8k"], "main", split="test")
    gsm_norm = {normalize_text(q) for q in gsm["question"]}
    log(f"GSM8K test questions: {len(gsm_norm)}")

    src = load_dataset(cfg["data"]["source"], split="train")
    log(f"MetaMathQA rows: {len(src)}")

    kept_idx, prompt_lens, resp_lens, ids_chunks = [], [], [], []
    excl = {"empty": 0, "gsm8k_test_match": 0, "too_long": 0}
    chunk = 2000
    pos = 0
    while len(kept_idx) < n_target and pos < len(src):
        rows = src[pos: pos + chunk]
        cand = []
        for j in range(len(rows["query"])):
            q, r = rows["query"][j], rows["response"][j]
            oq = rows.get("original_question", [None] * len(rows["query"]))[j]
            if not q or not q.strip() or not r or not r.strip():
                excl["empty"] += 1
                continue
            if normalize_text(q) in gsm_norm or (oq and normalize_text(oq) in gsm_norm):
                excl["gsm8k_test_match"] += 1
                continue
            cand.append((pos + j, q, r))
        prompts = [render(tok, [{"role": "user", "content": q}]) for _, q, _ in cand]
        p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
        r_ids = tok([r for _, _, r in cand], add_special_tokens=False)["input_ids"]
        for (i, _, _), p, rr in zip(cand, p_ids, r_ids):
            rr = rr + [eot]
            if len(p) + len(rr) > max_len:
                excl["too_long"] += 1
                continue
            kept_idx.append(i)
            prompt_lens.append(len(p))
            resp_lens.append(len(rr))
            ids_chunks.append(np.asarray(p + rr, dtype=np.int32))
            if len(kept_idx) == n_target:
                break
        pos += chunk
    assert len(kept_idx) == n_target, f"only {len(kept_idx)} eligible examples"

    lengths = np.array([len(x) for x in ids_chunks], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if limit is None else f"_limit{limit}"
    np.savez(DATA_DIR / f"train{tag}.npz", ids=np.concatenate(ids_chunks), offsets=offsets,
             prompt_len=np.array(prompt_lens, dtype=np.int32), resp_len=np.array(resp_lens, dtype=np.int32),
             source_index=np.array(kept_idx, dtype=np.int64))
    stats = {
        "n_selected": len(kept_idx),
        "last_source_index": int(kept_idx[-1]),
        "source_rows_scanned": int(kept_idx[-1]) + 1,
        "excluded": excl,
        "tokens_total": int(lengths.sum()),
        "response_tokens_total": int(np.sum(resp_lens)),
        "length_mean": float(lengths.mean()),
        "length_p50": float(np.median(lengths)),
        "length_p99": float(np.percentile(lengths, 99)),
        "length_max": int(lengths.max()),
        "example_prompt": render(tok, [{"role": "user", "content": "<question>"}]),
    }
    if limit is None:
        save_json(stats, RESULTS / "task1" / "data_stats.json")
        save_json({"metamathqa_source_indices": [int(i) for i in kept_idx]},
                  ROOT / "data" / "task1" / "train_source_indices.json", indent=None)
    log(f"data stats: {stats}")
    return stats


class TrainData:
    """Fixed-order access to the tokenised examples."""

    def __init__(self, path=None):
        path = path or DATA_DIR / "train.npz"
        z = np.load(path)
        self.ids, self.offsets = z["ids"], z["offsets"]
        self.prompt_len, self.resp_len = z["prompt_len"], z["resp_len"]
        self.n = len(self.prompt_len)

    def example(self, i: int) -> np.ndarray:
        return self.ids[self.offsets[i]: self.offsets[i + 1]]

    def collate(self, idxs, pad_id: int):
        import torch
        seqs = [self.example(i) for i in idxs]
        L = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        label_mask = torch.zeros((len(seqs), L), dtype=torch.bool)  # True where token is a target
        for b, (i, s) in enumerate(zip(idxs, seqs)):
            input_ids[b, : len(s)] = torch.from_numpy(s.astype("int64"))
            attn[b, : len(s)] = 1
            label_mask[b, self.prompt_len[i]: len(s)] = True  # response + end-of-turn
        return input_ids, attn, label_mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke test: select only N examples")
    build(ap.parse_args().limit)
