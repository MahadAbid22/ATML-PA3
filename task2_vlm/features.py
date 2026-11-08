"""Cache frozen SigLIP patch features (FP16) for every selected image.

google/siglip-base-patch16-224 -> last_hidden_state [196, 768] = the 14 x 14 patch grid in
row-major spatial order, after SigLIP's final (post) layer normalisation, applied exactly once
(verified numerically below). Deterministic preprocessing: the model's own image processor
(resize to 224 x 224, rescale, normalise); JPEG decoding with PIL; no augmentation.
The vision encoder exists only in this process, so it is never on the GPU during Qwen training.

Usage:  python -m task2_vlm.features [--limit N]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common.utils import OUTPUTS, RESULTS, ROOT, environment_info, gpu_peak_gb, load_json, load_yaml, log, save_json
from task2_vlm.download import image_path

CFG = ROOT / "configs" / "task2.yaml"
FEAT = OUTPUTS / "task2" / "features"


def main(limit: int | None = None, batch: int = 64):
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel
    cfg = load_yaml(CFG)
    lst = load_json(OUTPUTS / "task2" / "image_list.json")
    items = [(s, int(i)) for s in ("train2014", "val2014") for i in lst[s]]
    if limit:
        items = items[:limit]
    FEAT.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(cfg["vision"], revision=cfg["vision_revision"])
    try:
        enc = SiglipVisionModel.from_pretrained(cfg["vision"], revision=cfg["vision_revision"], dtype=torch.float16)
    except TypeError:
        enc = SiglipVisionModel.from_pretrained(cfg["vision"], revision=cfg["vision_revision"], torch_dtype=torch.float16)
    enc = enc.cuda().eval()
    load_s = time.perf_counter() - t0

    arr = np.lib.format.open_memmap(FEAT / "features.npy", mode="w+", dtype=np.float16, shape=(len(items), 196, 768))
    index = {f"{s}/{i}": k for k, (s, i) in enumerate(items)}
    ln_check = None
    t1 = time.perf_counter()
    decode_s = 0.0
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        td = time.perf_counter()
        ims = [Image.open(image_path(i, s)).convert("RGB") for s, i in chunk]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
        decode_s += time.perf_counter() - td
        with torch.no_grad():
            out = enc(pixel_values=px, output_hidden_states=(ln_check is None))
            feats = out.last_hidden_state
            if ln_check is None:  # final LayerNorm applied exactly once?
                pre = out.hidden_states[-1]
                ln = next(m for n, m in enc.named_modules() if n.endswith("post_layernorm"))
                post = ln(pre)
                twice = ln(post)
                ln_check = {"max_abs_diff_last_vs_post_ln(encoder_out)": float((post - feats).abs().max()),
                            "max_abs_diff_last_vs_double_ln": float((twice - feats).abs().max()),
                            "shape": list(feats.shape)}
                assert feats.shape[1:] == (196, 768), feats.shape
        arr[b:b + len(chunk)] = feats.float().cpu().numpy().astype(np.float16)
        if (b // batch) % 50 == 0:
            log(f"features {b + len(chunk)}/{len(items)}")
    arr.flush()
    torch.cuda.synchronize()
    save_json(index, FEAT / "index.json", indent=None)
    stats = {"n_images": len(items), "model_load_seconds": load_s, "encode_seconds": time.perf_counter() - t1,
             "decode_preprocess_seconds": decode_s, "feature_bytes": int(arr.nbytes), "dtype": "float16",
             "layernorm_check": ln_check, "gpu": gpu_peak_gb(), "environment": environment_info()}
    save_json(stats, RESULTS / "task2" / "feature_caching.json")
    log(f"cached {len(items)} images: {stats['encode_seconds']:.0f}s; LN check {ln_check}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.limit)
