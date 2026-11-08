"""Download COCO 2014 annotations, VQAv2 files and only the COCO images that are used.

Usage:  python -m task2_vlm.download --annotations      (annotation / question files)
        python -m task2_vlm.download --images           (images listed by task2_vlm.data)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
import urllib.request
import zipfile
from pathlib import Path

from common.utils import OUTPUTS, load_json, log

RAW = OUTPUTS / "task2" / "raw"
IMG = OUTPUTS / "task2" / "images"
COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VQA = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_FILES = ["v2_Questions_Train_mscoco.zip", "v2_Questions_Val_mscoco.zip", "v2_Annotations_Train_mscoco.zip",
             "v2_Annotations_Val_mscoco.zip", "v2_Complementary_Pairs_Train_mscoco.zip",
             "v2_Complementary_Pairs_Val_mscoco.zip"]


def _fetch(url: str, dest: Path, retries: int = 5) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(retries):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return dest
        except Exception as e:  # network hiccup: back off and retry
            if k == retries - 1:
                raise
            log(f"retry {url}: {e}")
            time.sleep(2 * (k + 1))
    return dest


def annotations():
    z = _fetch(COCO_ANN, RAW / "annotations_trainval2014.zip")
    with zipfile.ZipFile(z) as f:
        for name in ["annotations/captions_train2014.json", "annotations/captions_val2014.json",
                     "annotations/instances_train2014.json", "annotations/instances_val2014.json"]:
            if not (RAW / name).exists():
                f.extract(name, RAW)
    for fn in VQA_FILES:
        z = _fetch(VQA + fn, RAW / "vqa" / fn)
        with zipfile.ZipFile(z) as f:
            for name in f.namelist():
                if not (RAW / "vqa" / name).exists():
                    f.extract(name, RAW / "vqa")
    log("annotations ready")


def image_path(image_id: int, split: str) -> Path:
    return IMG / split / f"COCO_{split}_{image_id:012d}.jpg"


def images(workers: int = 32):
    lst = load_json(OUTPUTS / "task2" / "image_list.json")
    jobs = [(int(i), s) for s, ids in lst.items() for i in ids]
    todo = [(i, s) for i, s in jobs if not image_path(i, s).exists()]
    log(f"{len(jobs)} images listed, {len(todo)} to download")
    t0 = time.time()

    def get(job):
        i, s = job
        _fetch(f"http://images.cocodataset.org/{s}/COCO_{s}_{i:012d}.jpg", image_path(i, s))

    with cf.ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(get, todo), 1):
            if n % 1000 == 0:
                log(f"  {n}/{len(todo)} images")
    log(f"images ready in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", action="store_true")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.annotations:
        annotations()
    if a.images:
        images(a.workers)
