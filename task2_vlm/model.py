"""SigLIP-Qwen VLM: connector, input assembly, loss, teacher-forced NLL and greedy decoding.

Sequence layout (Qwen3 chat structure kept, thinking disabled):
  <|im_start|>user\\n [196 visual embeddings] \\n<question/instruction><|im_end|>\\n
  <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n <target><|im_end|>
The visual embeddings replace a placeholder inside the rendered user message, so the
template is the model's own. Supervision: target tokens + end-of-turn token. Visual tokens,
prompt and padding are masked. No-image inputs drop the visual embeddings entirely and the
attention mask and position ids are rebuilt for the shorter sequence.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.chat import end_of_turn_id, render

PLACEHOLDER = "<<<IMAGE>>>"


class Connector(nn.Module):
    """Linear 768 -> 1024, GELU, Linear 1024 -> 1024 (trainable, FP32)."""

    def __init__(self, d_vis=768, d_llm=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_vis, d_llm)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_llm, d_llm)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def pool_tokens(feats: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """196 -> 49 tokens by averaging each 2 x 2 block of the 14 x 14 grid (optional run)."""
    if n_tokens == 196:
        return feats
    B = feats.shape[0]
    g = feats.view(B, 14, 14, -1).permute(0, 3, 1, 2)
    g = F.avg_pool2d(g, 2)
    return g.permute(0, 2, 3, 1).reshape(B, 49, -1)


class PromptBuilder:
    """Caches token ids of the fixed template pieces."""

    def __init__(self, tok, prompts: dict, caps: dict):
        self.tok, self.prompts, self.caps = tok, prompts, caps
        self.eot = end_of_turn_id(tok)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self._cache = {}

    def instruction(self, ex: dict) -> str:
        if ex["kind"] == "caption":
            return self.prompts["caption"]
        return f"{ex['question']}\n{self.prompts['qa_suffix']}"

    def pieces(self, text: str):
        if text not in self._cache:
            r = render(self.tok, [{"role": "user", "content": PLACEHOLDER + "\n" + text}])
            pre, post = r.split(PLACEHOLDER)
            enc = lambda s: self.tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
            self._cache[text] = (enc(pre), enc(post))
        return self._cache[text]

    def target_ids(self, ex: dict, target: str | None = None) -> list[int]:
        t = target if target is not None else ex["answer"]
        ids = self.tok(t, add_special_tokens=False)["input_ids"]
        cap = self.caps["caption"] if ex["kind"] == "caption" else self.caps["answer"]
        return ids[:cap] + [self.eot]


class VLM(nn.Module):
    def __init__(self, llm, connector: Connector, builder: PromptBuilder, n_tokens: int, scale: float = 1.0):
        super().__init__()
        self.llm, self.connector, self.b, self.n_tokens = llm, connector, builder, n_tokens
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.embed = llm.get_input_embeddings()

    # ---------------------------------------------------------------- inputs
    def visual(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B, 196, 768] (FP16 cache) -> visual embeddings [B, n_tokens, 1024]."""
        x = pool_tokens(feats.float(), self.n_tokens)
        return self.connector(x) * self.scale

    def assemble(self, exs, feats, with_image, targets=None, left_pad=False, include_target=True):
        """Build inputs_embeds, attention mask, position ids and target mask for a batch.

        feats: [B, 196, 768] tensor (rows aligned with exs); with_image: list[bool].
        Returns dict; target_mask marks target tokens (incl. end-of-turn), eot_mask the end token.
        """
        dev = self.embed.weight.device
        vis = self.visual(feats.to(dev)) if any(with_image) else None
        seqs, tmask, emask = [], [], []
        for k, ex in enumerate(exs):
            pre, post = self.b.pieces(self.b.instruction(ex))
            tgt = self.b.target_ids(ex, None if targets is None else targets[k]) if include_target else []
            ids_pre = torch.tensor(pre, device=dev)
            ids_post = torch.tensor(post + tgt, device=dev)
            dt = self.embed.weight.dtype
            parts = [self.embed(ids_pre)]
            if with_image[k]:
                parts.append(vis[k].to(dt))
            parts.append(self.embed(ids_post))
            e = torch.cat([p.to(dt) for p in parts], 0)
            L = e.shape[0]
            tm = torch.zeros(L, dtype=torch.bool, device=dev)
            em = torch.zeros(L, dtype=torch.bool, device=dev)
            if include_target:
                tm[L - len(tgt):] = True
                em[L - 1] = True
            seqs.append(e)
            tmask.append(tm)
            emask.append(em)
        Lmax = max(s.shape[0] for s in seqs)
        D = seqs[0].shape[1]
        B = len(seqs)
        dtype = seqs[0].dtype
        emb = torch.zeros(B, Lmax, D, device=dev, dtype=dtype)
        attn = torch.zeros(B, Lmax, dtype=torch.long, device=dev)
        T = torch.zeros(B, Lmax, dtype=torch.bool, device=dev)
        E = torch.zeros(B, Lmax, dtype=torch.bool, device=dev)
        for i, s in enumerate(seqs):
            L = s.shape[0]
            sl = slice(Lmax - L, Lmax) if left_pad else slice(0, L)
            emb[i, sl] = s.to(dtype)
            attn[i, sl] = 1
            T[i, sl] = tmask[i]
            E[i, sl] = emask[i]
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        return {"inputs_embeds": emb, "attention_mask": attn, "position_ids": pos, "target_mask": T, "eot_mask": E}

    # ---------------------------------------------------------------- losses
    def example_losses(self, exs, feats, with_image, targets=None, exclude_eot=False):
        """Mean token NLL per example (teacher forcing). Returns tensor [B]."""
        batch = self.assemble(exs, feats, with_image, targets)
        tgt_ids = self._target_id_tensor(exs, targets, batch)
        out = self.llm.model(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                             position_ids=batch["position_ids"])
        h = out.last_hidden_state[:, :-1]
        m = batch["target_mask"][:, 1:].clone()
        if exclude_eot:
            m &= ~batch["eot_mask"][:, 1:]
        logits = self.llm.lm_head(h[m])
        nll = F.cross_entropy(logits.float(), tgt_ids[:, 1:][m], reduction="none")
        ex_idx = torch.arange(len(exs), device=nll.device)[:, None].expand_as(m)[m]
        sums = torch.zeros(len(exs), device=nll.device).index_add_(0, ex_idx, nll)
        cnt = torch.zeros(len(exs), device=nll.device).index_add_(0, ex_idx, torch.ones_like(nll))
        return sums / cnt.clamp(min=1)

    def _target_id_tensor(self, exs, targets, batch):
        B, L = batch["target_mask"].shape
        ids = torch.full((B, L), -100, dtype=torch.long, device=batch["target_mask"].device)
        for k, ex in enumerate(exs):
            tgt = self.b.target_ids(ex, None if targets is None else targets[k])
            pos = batch["target_mask"][k].nonzero().squeeze(-1)
            ids[k, pos] = torch.tensor(tgt, device=ids.device)
        return ids

    # ---------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, exs, feats, with_image, max_new_tokens=16):
        """Greedy decoding with KV cache; stops at <|im_end|> / <|endoftext|>."""
        batch = self.assemble(exs, feats, with_image, left_pad=True, include_target=False)
        emb, attn, pos = batch["inputs_embeds"], batch["attention_mask"], batch["position_ids"]
        stop = {self.b.eot, self.b.tok.eos_token_id}
        out = self.llm(inputs_embeds=emb.to(self.llm.dtype), attention_mask=attn, position_ids=pos, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        B = len(exs)
        gen = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=nxt.device)
        last_pos = pos[:, -1]
        for step in range(max_new_tokens):
            for i in range(B):
                if not done[i]:
                    t = int(nxt[i])
                    if t in stop:
                        done[i] = True
                    else:
                        gen[i].append(t)
            if bool(done.all()) or step == max_new_tokens - 1:
                break
            attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=attn.device)], 1)
            last_pos = last_pos + 1
            out = self.llm(input_ids=nxt[:, None], attention_mask=attn, position_ids=last_pos[:, None],
                           past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
        texts = [self.b.tok.decode(g, skip_special_tokens=True).strip() for g in gen]
        return texts, [len(g) for g in gen], [bool(d) for d in done.tolist()]


class FeatureStore:
    """Memory-mapped FP16 feature cache keyed by (split, image_id)."""

    def __init__(self, path_npy, index: dict):
        self.arr = np.load(path_npy, mmap_mode="r")
        self.index = index

    def get(self, image_ids_with_split) -> torch.Tensor:
        rows = [self.index[f"{s}/{i}"] for s, i in image_ids_with_split]
        return torch.from_numpy(np.ascontiguousarray(self.arr[rows]))
