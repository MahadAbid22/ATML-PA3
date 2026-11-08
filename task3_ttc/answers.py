"""Answer extraction, normalisation, grouping and grading for Task 3.

MATH: answers are grouped with math-verify equivalence (checked in both directions) without
consulting the gold answer; a group is correct iff its representative (first-seen member) is
math-verify-equivalent to the gold answer. MathVista: the released testmini metadata gives the
question type, answer type, choices and precision; predictions are normalised with MathVista's
rules (multiple choice -> option letter via letter / exact text / most similar choice; integer ->
int; float -> rounded to the stated precision; list -> string) and grouped by exact equality of the
normalised form. Missing (null/empty) extractions form one NO_ANSWER group, always incorrect.
"""
from __future__ import annotations

import logging
import re
import signal
import sys

NO_ANSWER = "NO_ANSWER"
# math-verify warns on every call that its own timeout is disabled; the SIGALRM guard below replaces it
logging.getLogger("math_verify").setLevel(logging.ERROR)


# ---------------------------------------------------------------- boxed extraction (README rule)
def last_boxed(text: str) -> str | None:
    """Content of the last literal '\\boxed{' with balanced braces; None if absent/empty/unclosed."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth, k = 1, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                s = text[j:k].strip()
                return s or None
        k += 1
    return None


# ---------------------------------------------------------------- MATH (math-verify)
class _Timeout(Exception):
    pass


def _with_timeout(fn, seconds=5):
    if sys.platform == "win32" or seconds is None:
        return fn()

    def handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def mv_parse(s: str):
    from math_verify import parse
    try:
        return _with_timeout(lambda: parse("$\\boxed{" + s + "}$", parsing_timeout=None))
    except Exception:
        return []


def mv_equal(pa, pb) -> bool:
    from math_verify import verify
    if not pa or not pb:
        return False
    try:
        return bool(_with_timeout(lambda: verify(pa, pb, timeout_seconds=None) or verify(pb, pa, timeout_seconds=None)))
    except Exception:
        return False


def group_math(extracted: list[str | None], gold: str) -> tuple[list[str], dict[str, bool]]:
    """Return per-rollout group labels and {label: correct} for one question."""
    labels, reps, parsed = [], [], {}
    for a in extracted:
        a = a.strip() if isinstance(a, str) else None
        if not a:
            labels.append(NO_ANSWER)
            continue
        if a in parsed:  # identical string -> same group as before
            labels.append(parsed[a])
            continue
        pa = mv_parse(a)
        hit = None
        for rep, prep in reps:
            if a == rep or mv_equal(prep, pa):
                hit = rep
                break
        if hit is None:
            reps.append((a, pa))
            hit = a
        parsed[a] = hit
        labels.append(hit)
    pg = mv_parse(gold)
    correct = {NO_ANSWER: False}
    for rep, prep in reps:
        correct[rep] = mv_equal(pg, prep)
    return labels, correct


# ---------------------------------------------------------------- MathVista rules
def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _clean_number(s: str) -> str:
    """Strip LaTeX wrappers / units so float() can read e.g. '0.0115 \\, \\text{m}' -> '0.0115'."""
    s = re.sub(r"\\text\{[^}]*\}|\\mathrm\{[^}]*\}|\\,|\\!|\\;|\\%|%|\$|\\\(|\\\)|\^\\circ|°", " ", s)
    s = s.replace(",", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", s)
    return m.group(0) if m else s


def mathvista_normalize(extraction: str | None, meta: dict) -> str:
    """MathVista normalize_extracted_answer, with option letters for multiple choice."""
    if extraction is None or (isinstance(extraction, str) and not extraction.strip()):
        return NO_ANSWER
    ext = str(extraction).strip()
    qtype, atype = meta["question_type"], meta["answer_type"]
    if qtype == "multi_choice":
        choices = [str(c) for c in meta["choices"]]
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        m = re.findall(r"\(([a-zA-Z])\)", ext)
        cand = m[0].upper() if m else re.sub(r"[^A-Za-z]", "", ext).upper() if len(ext) <= 3 else ext
        if cand in letters:
            return cand
        if ext in choices:
            return letters[choices.index(ext)]
        dists = [_levenshtein(ext.lower(), c.lower()) for c in choices]
        return letters[dists.index(min(dists))]
    if atype == "integer":
        try:
            return str(int(float(_clean_number(ext))))
        except Exception:
            return "UNPARSED:" + ext
    if atype == "float":
        try:
            return str(round(float(_clean_number(ext)), int(meta.get("precision") or 0)))
        except Exception:
            return "UNPARSED:" + ext
    return ext  # list / text: exact string


def mathvista_gold(meta: dict) -> str:
    if meta["question_type"] == "multi_choice":
        choices = [str(c) for c in meta["choices"]]
        return chr(ord("A") + choices.index(str(meta["answer"])))
    return mathvista_normalize(str(meta["answer"]), meta)


def group_mathvista(extracted: list[str | None], meta: dict) -> tuple[list[str], dict[str, bool]]:
    labels = [mathvista_normalize(a, meta) for a in extracted]
    gold = mathvista_gold(meta)
    correct = {l: (l == gold and l != NO_ANSWER) for l in set(labels)}
    correct[NO_ANSWER] = False
    return labels, correct
