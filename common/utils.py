"""Shared helpers: seeding, IO, timing, GPU memory, environment capture."""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
# Large artefacts (checkpoints, merged models, caches) live outside version control.
OUTPUTS = Path(os.environ.get("PA3_OUTPUTS", ROOT / "outputs"))


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path, indent=2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, path, mode="w") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


@contextmanager
def timer(store: dict, key: str, cuda_sync: bool = True):
    """Wall-clock timer; synchronises CUDA so GPU work is fully counted."""
    import torch
    if cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def gpu_peak_gb(reset: bool = False) -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    out = {
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
    }
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return out


def dir_size_bytes(path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["torch", "transformers", "datasets", "lm_eval", "math_verify", "numpy", "pandas", "pyarrow"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hf_snapshot(repo_id: str, revision: str | None = None, repo_type: str = "model") -> tuple[str, str]:
    """Download (or reuse) a Hugging Face snapshot; return (local_path, resolved_commit)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id, revision=revision, repo_type=repo_type)
    commit = Path(path).name  # snapshots/<commit>
    return path, commit
