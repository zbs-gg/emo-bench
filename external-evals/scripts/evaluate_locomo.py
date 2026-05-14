"""LOCOMO evaluator — F1 + adversarial refusal rate per official scoring rubric.

Adapted from locomo/task_eval/evaluation.py (Maharana et al., ACL 2024) to work
on jsonl output of run_pulse_locomo.py. Pure-python, no nltk/bert_score dependency.

Category semantics (from paper Table 2):
  1: multi-hop reasoning (split answer on ';', mean of max-F1)
  2: temporal reasoning (direct F1)
  3: open-domain knowledge (direct F1; answer first ';' segment only)
  4: single-hop (direct F1)
  5: adversarial (checks that hypothesis says "don't know"-ish — expected refusal)

Usage:
  python evaluate_locomo.py --hyps path/to/hyps.jsonl --out path/to/scored.json
"""
from __future__ import annotations
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
import sys
import unicodedata


def _normalize(text):
    """Lowercase, strip punctuation and articles."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    # remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # remove articles + common stops
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between normalized prediction and ground truth."""
    p_toks = _normalize(prediction).split()
    g_toks = _normalize(ground_truth).split()
    if not p_toks or not g_toks:
        return 0.0 if p_toks != g_toks else 1.0
    common = Counter(p_toks) & Counter(g_toks)
    n_overlap = sum(common.values())
    if n_overlap == 0:
        return 0.0
    precision = n_overlap / len(p_toks)
    recall = n_overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _multi_hop_f1(prediction: str, ground_truth: str) -> float:
    """Split ground truth on ';' → mean of per-sub-answer max-F1 against prediction."""
    subs = [s.strip() for s in ground_truth.split(";") if s.strip()]
    if not subs:
        return _f1_score(prediction, ground_truth)
    # Each sub-answer: max F1 across single prediction (we don't have multi-pred here, one hyp)
    return sum(_f1_score(prediction, sub) for sub in subs) / len(subs)


ADVERSARIAL_REFUSAL_PATTERNS = (
    "no information available",
    "not mentioned",
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "cannot determine",
    "not available",
    "not provided",
    "not specified",
    "no information",
    "isn't mentioned",
    "is not available",
    "insufficient",
    "not found",
)


def _is_adversarial_refusal(hyp: str) -> bool:
    h = hyp.lower() if hyp else ""
    return any(p in h for p in ADVERSARIAL_REFUSAL_PATTERNS)


def score_one(row: dict) -> tuple[float, int]:
    """Return (score 0-1, category). Unknown category returns (0, 0)."""
    cat = int(row.get("category") or 0)
    hyp = row.get("hypothesis", "") or ""
    if cat == 5:
        # Adversarial: refusal is the correct answer
        return (1.0 if _is_adversarial_refusal(hyp) else 0.0), cat
    answer = row.get("answer", "") or ""
    # LoCoMo answers are sometimes ints (years, counts) — normalize to str
    if not isinstance(answer, str):
        answer = str(answer)
    if cat == 3:
        # Take first ';'-segment only (per official evaluator)
        answer = answer.split(";")[0].strip()
    if cat == 1:
        return _multi_hop_f1(hyp, answer), cat
    if cat in (2, 3, 4):
        return _f1_score(hyp, answer), cat
    # Default: treat as F1
    return _f1_score(hyp, answer), cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    by_cat: dict[int, list[float]] = {}
    by_sample: dict[str, list[float]] = {}
    n_refusal_true = 0
    n_refusal_false = 0
    n_total = 0

    scored_rows = []
    with args.hyps.open() as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            score, cat = score_one(row)
            row["_score"] = score
            row["_category"] = cat
            scored_rows.append(row)
            by_cat.setdefault(cat, []).append(score)
            sid = row.get("sample_id", "")
            by_sample.setdefault(sid, []).append(score)
            n_total += 1
            if cat == 5:
                if _is_adversarial_refusal(row.get("hypothesis", "")):
                    n_refusal_true += 1
                else:
                    n_refusal_false += 1

    def _mean(xs): return round(sum(xs) / len(xs), 4) if xs else 0.0

    summary = {
        "n_total": n_total,
        "overall_mean": _mean([s for scores in by_cat.values() for s in scores]),
        "by_category": {
            "1_multi_hop":   {"n": len(by_cat.get(1, [])), "mean_f1": _mean(by_cat.get(1, []))},
            "2_temporal":    {"n": len(by_cat.get(2, [])), "mean_f1": _mean(by_cat.get(2, []))},
            "3_open_domain": {"n": len(by_cat.get(3, [])), "mean_f1": _mean(by_cat.get(3, []))},
            "4_single_hop":  {"n": len(by_cat.get(4, [])), "mean_f1": _mean(by_cat.get(4, []))},
            "5_adversarial": {"n": len(by_cat.get(5, [])),
                              "refusal_accuracy": _mean(by_cat.get(5, [])),
                              "refused": n_refusal_true, "answered": n_refusal_false},
        },
        "by_sample": {sid: _mean(scores) for sid, scores in by_sample.items()},
    }

    print(f"LOCOMO evaluation — hyps: {args.hyps.name}")
    print(f"Total questions: {n_total}")
    print(f"Overall mean score: {summary['overall_mean']:.4f}")
    print("\nPer category:")
    for k, v in summary["by_category"].items():
        print(f"  {k:<18} n={v['n']:>4} mean_f1={v.get('mean_f1', v.get('refusal_accuracy', 0)):.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": scored_rows},
                                        ensure_ascii=False, indent=2))
        print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
