"""LoRA and OPLoRA adapters (own implementation, shared by Tasks 1 and 2).

LoRA (Hu et al., 2021):      W = W0 + (alpha / r) * B A,
    A in R^{r x d_in}  (Kaiming-uniform, a = sqrt(5), as in PEFT),
    B in R^{d_out x r} (zeros), so the adapted model starts exactly at W0.

OPLoRA (Xiong & Xie): with W0 = U S V^T and frozen leading bases U_k, V_k,
    B~ = B - U_k (U_k^T B),   A~ = A - (A V_k) V_k^T,   W = W0 + (alpha / r) * B~ A~.
The effective (projected) factors are formed from the trainable factors on every
forward pass; dense d x d projection matrices are never built. Hence
U_k^T dW = 0 and dW V_k = 0 by construction.

A single implementation is used for LoRA and OPLoRA so that, at matched rank and
seed, both start from bit-identical unprojected factors.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float,
                 generator: torch.Generator, U_k: torch.Tensor | None = None,
                 V_k: torch.Tensor | None = None):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        d_out, d_in = base.weight.shape
        dev = base.weight.device
        A = torch.empty(r, d_in, dtype=torch.float32)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=generator)  # PEFT default for A
        self.lora_A = nn.Parameter(A.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.float32, device=dev))
        self.protected = U_k is not None
        if self.protected:
            self.register_buffer("U_k", U_k.to(dev, torch.float32).contiguous(), persistent=False)
            self.register_buffer("V_k", V_k.to(dev, torch.float32).contiguous(), persistent=False)

    def effective_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (A~, B~) in FP32. For plain LoRA these are (A, B)."""
        A, B = self.lora_A, self.lora_B
        if self.protected:
            A = A - (A @ self.V_k) @ self.V_k.T
            B = B - self.U_k @ (self.U_k.T @ B)
        return A, B

    def delta_weight(self) -> torch.Tensor:
        A, B = self.effective_factors()
        return self.scaling * (B @ A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            A, B = self.effective_factors()  # projections always in FP32
        if not torch.is_autocast_enabled(x.device.type):
            A, B = A.to(x.dtype), B.to(x.dtype)
        h = self.lora_dropout(x)
        return y + F.linear(F.linear(h, A), B) * self.scaling


def _get_parent(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def target_linear_names(model: nn.Module, targets: list[str]) -> list[str]:
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    return names  # named_modules order is deterministic (layer by layer)


@torch.no_grad()
def top_k_bases(W: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Leading left/right singular bases of W (d_out x d_in), computed in FP32."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return U[:, :k].contiguous(), Vh[:k].T.contiguous()


def inject_adapters(model: nn.Module, targets: list[str], r: int, alpha: float, dropout: float,
                    seed: int, protect_k: int | None = None) -> dict:
    """Freeze nothing here; replace each target nn.Linear by a LoRALinear.

    Returns overhead statistics (basis computation time and storage for OPLoRA).
    """
    gen = torch.Generator().manual_seed(seed)
    stats = {"n_adapted_modules": 0, "basis_seconds": 0.0, "basis_bytes": 0}
    for name in target_linear_names(model, targets):
        parent, attr = _get_parent(model, name)
        base = getattr(parent, attr)
        U_k = V_k = None
        if protect_k:
            if base.weight.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            U_k, V_k = top_k_bases(base.weight.data, protect_k)
            if base.weight.is_cuda:
                torch.cuda.synchronize()
            stats["basis_seconds"] += time.perf_counter() - t0
            stats["basis_bytes"] += (U_k.numel() + V_k.numel()) * 4
        setattr(parent, attr, LoRALinear(base, r, alpha, dropout, gen, U_k, V_k))
        stats["n_adapted_modules"] += 1
    return stats


def lora_modules(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            yield name, m


def adapter_state_dict(model: nn.Module, projected: bool = False) -> dict:
    """Adapter factors on CPU (FP32). projected=True exports A~, B~ (OPLoRA export)."""
    sd = {}
    for name, m in lora_modules(model):
        if projected:
            with torch.no_grad():
                A, B = m.effective_factors()
        else:
            A, B = m.lora_A, m.lora_B
        sd[f"{name}.lora_A"] = A.detach().float().cpu().contiguous()
        sd[f"{name}.lora_B"] = B.detach().float().cpu().contiguous()
    return sd


def load_adapter_factors(model: nn.Module, sd: dict) -> None:
    with torch.no_grad():
        for name, m in lora_modules(model):
            m.lora_A.copy_(sd[f"{name}.lora_A"])
            m.lora_B.copy_(sd[f"{name}.lora_B"])


@torch.no_grad()
def merge_factors_into(model: nn.Module, factors: dict, scaling: float) -> None:
    """Merge exported factors into a plain (un-adapted) model: W <- W + s * B A (FP32 maths)."""
    modules = dict(model.named_modules())
    names = sorted({k.rsplit(".lora_", 1)[0] for k in factors})
    for name in names:
        lin = modules[name]
        A = factors[f"{name}.lora_A"].to(lin.weight.device, torch.float32)
        B = factors[f"{name}.lora_B"].to(lin.weight.device, torch.float32)
        W = lin.weight.data.float() + scaling * (B @ A)
        lin.weight.data.copy_(W.to(lin.weight.dtype))


def count_trainable(model: nn.Module) -> int:
    seen, n = set(), 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            n += p.numel()
    return n
