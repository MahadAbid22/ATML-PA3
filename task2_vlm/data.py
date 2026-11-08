"""Task 2 data selection (seed 6304). Writes example files and the saved ID lists.

Components
* COCO captions: 6,000 train2014 images x 3 captions (training, replay); 96 val2014 images x 1 (dev).
* Annotation-derived grounding (programmatic, from COCO instances):
    presence  "Does this image contain <object>?"  image with the category vs image without it (yes/no)
    counting  "How many <objects> can you see?"     two images with different counts in [1, 4]
  Training pools are built on training images only; dev/test pairs on val2014 images.
* VQAv2 complementary pairs: both questions have a normalised consensus answer given by >= 8/10
  annotators, the two consensus answers differ; target = most frequent original answer string in
  the consensus group; all ten human answers kept for scoring.
Every development and final-test suite is image-disjoint from every other suite and from training
(train2014 vs val2014), which is asserted below.

Usage:  python -m task2_vlm.data
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from common.utils import OUTPUTS, RESULTS, ROOT, load_yaml, log, save_json
from task2_vlm.download import RAW
from task2_vlm.vqa_norm import normalize_answer

CFG = ROOT / "configs" / "task2.yaml"
OUT = OUTPUTS / "task2"
IDS = ROOT / "data" / "task2"

PLURAL = {"person": "people", "bus": "buses", "skis": "skis", "scissors": "scissors", "sheep": "sheep",
          "mouse": "mice", "knife": "knives", "wine glass": "wine glasses", "couch": "couches",
          "bench": "benches", "sandwich": "sandwiches", "toothbrush": "toothbrushes", "glass": "glasses",
          "dining table": "dining tables", "hot dog": "hot dogs", "teddy bear": "teddy bears",
          "skateboard": "skateboards", "surfboard": "surfboards", "snowboard": "snowboards",
          "potted plant": "potted plants", "hair drier": "hair driers", "cell phone": "cell phones",
          "traffic light": "traffic lights", "fire hydrant": "fire hydrants", "stop sign": "stop signs",
          "parking meter": "parking meters", "sports ball": "sports balls", "baseball bat": "baseball bats",
          "baseball glove": "baseball gloves", "tennis racket": "tennis rackets", "giraffe": "giraffes",
          "broccoli": "broccoli"}
MASS = {"skis", "scissors", "broccoli"}


def plural(name: str) -> str:
    return PLURAL.get(name, name + "s")


def with_article(name: str) -> str:
    if name in MASS:
        return f"any {name}"
    return ("an " if name[0] in "aeiou" else "a ") + name


def presence_q(name: str) -> str:
    return f"Does this image contain {with_article(name)}?"


def counting_q(name: str) -> str:
    return f"How many {plural(name)} can you see?"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def instance_index(path):
    """image_id -> {cat_id: {"n": non-crowd count, "crowd": bool, "areas": [...]}}, plus image areas."""
    d = _load(path)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    area = {im["id"]: im["width"] * im["height"] for im in d["images"]}
    idx = defaultdict(dict)
    for a in d["annotations"]:
        e = idx[a["image_id"]].setdefault(a["category_id"], {"n": 0, "crowd": False, "areas": []})
        if a.get("iscrowd", 0):
            e["crowd"] = True
        else:
            e["n"] += 1
            e["areas"].append(a["area"])
    images = sorted(area)
    del d
    return cats, area, idx, images


def grounding_candidates(cats, area, idx, image_ids, gcfg):
    """Per category: positive images (presence), images without it, and count -> images (counting)."""
    pos, neg, cnt = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    lo, hi = gcfg["count_range"]
    for im in image_ids:
        A = area[im]
        anns = idx.get(im, {})
        for c in cats:
            e = anns.get(c)
            if e is None:
                neg[c].append(im)
                continue
            if e["areas"] and max(e["areas"]) >= gcfg["presence_min_area"] * A:
                pos[c].append(im)
            if not e["crowd"] and lo <= e["n"] <= hi and min(e["areas"]) >= gcfg["count_min_area"] * A:
                cnt[c][e["n"]].append(im)
    return pos, neg, cnt


def sample_presence(rng, cats, pos, neg, n, used=None, min_each=3):
    exclusive = used is not None  # evaluation suites: every image used at most once
    used = used if exclusive else set()
    ok = sorted(c for c in cats if len(pos[c]) >= min_each and len(neg[c]) >= min_each)
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        c = ok[rng.integers(len(ok))]
        a = int(pos[c][rng.integers(len(pos[c]))])
        b = int(neg[c][rng.integers(len(neg[c]))])
        if a in used or b in used or (c, a, b) in seen:
            continue
        seen.add((c, a, b))
        if exclusive:
            used.update((a, b))
        q = presence_q(cats[c])
        out.append({"kind": "presence", "category": cats[c], "question": q,
                    "a": {"image_id": a, "answer": "yes"}, "b": {"image_id": b, "answer": "no"}})
    assert len(out) == n, f"presence: only {len(out)}/{n}"
    return out


def sample_counting(rng, cats, cnt, n, used=None):
    exclusive = used is not None
    used = used if exclusive else set()
    ok = sorted(c for c in cats if sum(1 for k in cnt[c] if cnt[c][k]) >= 2)
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        c = ok[rng.integers(len(ok))]
        counts = sorted(k for k in cnt[c] if cnt[c][k])
        ka, kb = rng.choice(counts, 2, replace=False)
        a = int(cnt[c][ka][rng.integers(len(cnt[c][ka]))])
        b = int(cnt[c][kb][rng.integers(len(cnt[c][kb]))])
        if a == b or a in used or b in used or (c, a, b) in seen:
            continue
        seen.add((c, a, b))
        if exclusive:
            used.update((a, b))
        out.append({"kind": "counting", "category": cats[c], "question": counting_q(cats[c]),
                    "a": {"image_id": a, "answer": str(int(ka))}, "b": {"image_id": b, "answer": str(int(kb))}})
    assert len(out) == n, f"counting: only {len(out)}/{n}"
    return out


def vqa_candidates(split: str, tok, caps, min_agree):
    tag = "train2014" if split == "train" else "val2014"
    qs = {q["question_id"]: q for q in _load(RAW / "vqa" / f"v2_OpenEnded_mscoco_{tag}_questions.json")["questions"]}
    anns = {a["question_id"]: a for a in _load(RAW / "vqa" / f"v2_mscoco_{tag}_annotations.json")["annotations"]}
    pairs = _load(RAW / "vqa" / f"v2_mscoco_{tag}_complementary_pairs.json")

    def consensus(qid):
        raw = [x["answer"] for x in anns[qid]["answers"]]
        norm = [normalize_answer(x) for x in raw]
        top, k = Counter(norm).most_common(1)[0]
        if k < min_agree:
            return None
        group = [r for r, n in zip(raw, norm) if n == top]
        target = Counter(group).most_common(1)[0][0]  # most frequent original string (ties: first seen)
        return {"canonical": top, "target": target, "answers": raw, "agree": k}

    out = defaultdict(list)
    for qa, qb in pairs:
        if qa not in anns or qb not in anns:
            continue
        ca, cb = consensus(qa), consensus(qb)
        if ca is None or cb is None or ca["canonical"] == cb["canonical"]:
            continue
        ta, tb = anns[qa]["answer_type"], anns[qb]["answer_type"]
        if ta != tb:
            continue
        q = qs[qa]["question"]
        if len(tok(q, add_special_tokens=False)["input_ids"]) > caps["question"]:
            continue
        if max(len(tok(c["target"], add_special_tokens=False)["input_ids"]) for c in (ca, cb)) > caps["answer"]:
            continue
        ia, ib = qs[qa]["image_id"], qs[qb]["image_id"]
        if ia == ib:
            continue
        out[ta].append({"kind": "vqa", "answer_type": ta, "question": q, "pair_id": f"{qa}-{qb}",
                        "a": {"image_id": ia, "question_id": qa, "answer": ca["target"],
                              "canonical": ca["canonical"], "answers": ca["answers"]},
                        "b": {"image_id": ib, "question_id": qb, "answer": cb["target"],
                              "canonical": cb["canonical"], "answers": cb["answers"]}})
    for t in out:
        out[t].sort(key=lambda p: p["pair_id"])
    return out


def pick_pairs(rng, cands_by_type, counts, used=None):
    sel = []
    for t, n in counts.items():
        c = cands_by_type[t]
        order = rng.permutation(len(c))
        got = 0
        for i in order:
            p = c[i]
            if used is not None and (p["a"]["image_id"] in used or p["b"]["image_id"] in used):
                continue
            sel.append(p)
            if used is not None:
                used.update((p["a"]["image_id"], p["b"]["image_id"]))
            got += 1
            if got == n:
                break
        assert got == n, f"vqa {t}: only {got}/{n}"
    return sel


def main():
    from transformers import AutoTokenizer
    cfg = load_yaml(CFG)
    dc, seed = cfg["data"], cfg["seed"]
    tok = AutoTokenizer.from_pretrained(cfg["llm"], revision=cfg["llm_revision"])
    rng = np.random.default_rng(seed)

    # ---------------- VQAv2
    vtr = vqa_candidates("train", tok, cfg["caps"], dc["vqa"]["min_agree"])
    vva = vqa_candidates("val", tok, cfg["caps"], dc["vqa"]["min_agree"])
    log("VQA candidate pairs train " + str({k: len(v) for k, v in vtr.items()}) +
        " val " + str({k: len(v) for k, v in vva.items()}))
    vqa_train = pick_pairs(rng, vtr, dc["vqa"]["train"])
    used_val: set[int] = set()
    vqa_test = pick_pairs(rng, vva, dc["vqa"]["test"], used_val)
    vqa_dev = pick_pairs(rng, vva, dc["vqa"]["dev"], used_val)

    # ---------------- captions (train)
    cap_tr = _load(RAW / "annotations" / "captions_train2014.json")
    by_img = defaultdict(list)
    for a in cap_tr["annotations"]:
        by_img[a["image_id"]].append(a)
    del cap_tr
    tr_imgs = sorted(i for i, v in by_img.items() if len(v) >= dc["captions_per_train_image"])
    cap_imgs = sorted(int(i) for i in rng.choice(tr_imgs, dc["caption_train_images"], replace=False))
    captions_train = []
    for im in cap_imgs:
        anns = sorted(by_img[im], key=lambda a: a["id"])
        pick = sorted(rng.choice(len(anns), dc["captions_per_train_image"], replace=False))
        for j in pick:
            captions_train.append({"kind": "caption", "image_id": im, "caption_id": anns[j]["id"],
                                   "answer": anns[j]["caption"].strip()})
    del by_img

    # ---------------- grounding: training pools on training images only
    train_images = sorted(set(cap_imgs) | {p[s]["image_id"] for p in vqa_train for s in "ab"})
    cats, area, idx, _ = instance_index(RAW / "annotations" / "instances_train2014.json")
    pos, neg, cnt = grounding_candidates(cats, area, idx, train_images, dc["grounding"])
    del idx
    gp = dc["grounding"]["train_pool"]
    pres_pool = sample_presence(rng, cats, pos, neg, gp["presence"])
    count_pool = sample_counting(rng, cats, cnt, gp["counting"])

    # ---------------- grounding: dev/test on val2014 images not used by VQA suites
    cats_v, area_v, idx_v, val_images = instance_index(RAW / "annotations" / "instances_val2014.json")
    val_free = [i for i in val_images if i not in used_val]
    pos_v, neg_v, cnt_v = grounding_candidates(cats_v, area_v, idx_v, val_free, dc["grounding"])
    del idx_v
    g_test = sample_presence(rng, cats_v, pos_v, neg_v, dc["grounding"]["test_pairs"]["presence"], used_val) + \
        sample_counting(rng, cats_v, cnt_v, dc["grounding"]["test_pairs"]["counting"], used_val)
    g_dev = sample_presence(rng, cats_v, pos_v, neg_v, dc["grounding"]["dev_pairs"]["presence"], used_val) + \
        sample_counting(rng, cats_v, cnt_v, dc["grounding"]["dev_pairs"]["counting"], used_val)

    # ---------------- caption dev on remaining val images
    cap_va = _load(RAW / "annotations" / "captions_val2014.json")
    by_img_v = defaultdict(list)
    for a in cap_va["annotations"]:
        by_img_v[a["image_id"]].append(a)
    cand = sorted(i for i in by_img_v if i not in used_val)
    dev_imgs = sorted(int(i) for i in rng.choice(cand, dc["caption_dev_images"], replace=False))
    captions_dev = []
    for im in dev_imgs:
        anns = sorted(by_img_v[im], key=lambda a: a["id"])
        j = int(rng.integers(len(anns)))
        captions_dev.append({"kind": "caption", "image_id": im, "caption_id": anns[j]["id"],
                             "answer": anns[j]["caption"].strip()})
        used_val.add(im)

    # ---------------- assign pair ids, isolation checks, save
    for name, lst in (("presence_pool", pres_pool), ("counting_pool", count_pool), ("grounding_test", g_test),
                      ("grounding_dev", g_dev)):
        for k, p in enumerate(lst):
            p["pair_id"] = f"{name}-{k}"

    def imgs(pairs):
        return {p[s]["image_id"] for p in pairs for s in "ab"}
    suites = {"caption_dev": {c["image_id"] for c in captions_dev}, "grounding_dev": imgs(g_dev),
              "grounding_test": imgs(g_test), "vqa_dev": imgs(vqa_dev), "vqa_test": imgs(vqa_test)}
    names = list(suites)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (suites[names[i]] & suites[names[j]]), f"overlap {names[i]} / {names[j]}"
    train_set = set(train_images)
    for n in names:
        assert not (suites[n] & train_set), f"eval suite {n} overlaps training"  # also different splits
    isolation = {"suites_pairwise_disjoint": True, "eval_disjoint_from_training": True,
                 "eval_images": {k: len(v) for k, v in suites.items()},
                 "train_images": len(train_set), "train_split": "train2014", "eval_split": "val2014"}

    examples = {"captions_train": captions_train, "captions_dev": captions_dev, "presence_pool": pres_pool,
                "counting_pool": count_pool, "vqa_train": vqa_train, "grounding_dev": g_dev,
                "grounding_test": g_test, "vqa_dev": vqa_dev, "vqa_test": vqa_test}
    save_json(examples, OUT / "examples.json", indent=None)
    save_json({"train2014": sorted(train_set), "val2014": sorted(set().union(*suites.values()))},
              OUT / "image_list.json", indent=None)

    def ids_only(p):
        if p["kind"] == "caption":
            return {"image_id": p["image_id"], "caption_id": p["caption_id"]}
        d = {"pair_id": p["pair_id"], "kind": p["kind"], "image_a": p["a"]["image_id"], "image_b": p["b"]["image_id"]}
        if p["kind"] == "vqa":
            d.update(question_a=p["a"]["question_id"], question_b=p["b"]["question_id"], answer_type=p["answer_type"])
        else:
            d.update(category=p["category"], answer_a=p["a"]["answer"], answer_b=p["b"]["answer"])
        return d
    save_json({k: [ids_only(x) for x in v] for k, v in examples.items()} | {"seed": seed},
              IDS / "selection_ids.json", indent=None)
    stats = {k: len(v) for k, v in examples.items()} | {"isolation": isolation,
             "vqa_candidate_pairs": {"train": {k: len(v) for k, v in vtr.items()},
                                     "val": {k: len(v) for k, v in vva.items()}},
             "grounding_train_categories_presence": len({p["category"] for p in pres_pool}),
             "grounding_train_categories_counting": len({p["category"] for p in count_pool}),
             "counting_answer_hist_train": dict(Counter(p[s]["answer"] for p in count_pool for s in "ab"))}
    save_json(stats, RESULTS / "task2" / "data_stats.json")
    log(f"task2 data: {json.dumps({k: len(v) for k, v in examples.items()})}")


if __name__ == "__main__":
    main()
