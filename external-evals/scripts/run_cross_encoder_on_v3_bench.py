"""Cross-encoder baseline for empathic-memory-bench-v3.

Purpose: state-conditioned cross-encoder vs Pulse v3 state-conditional ranking.
This is the peer-review comparator that the 'delta' reviewer called missing.

Model: BAAI/bge-reranker-v2-m3 (MIT license, multilingual, ~600M params).
Scoring: for each test, 60 (query_string, event_text) pairs scored jointly.
State projection: matches cosine_state baseline — query | mood: ...; hrv: ...

Install (isolated venv to avoid polluting bench env):
  python3 -m venv .venv-cross-encoder
  .venv-cross-encoder/bin/pip install sentence-transformers torch

Output JSON: matches Mem0/Graphiti adapter schema.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# ─── State serialisation ──────────────────────────────────────────────────────
# Matches baselines_state_aware.py exactly so comparisons are apples-to-apples.

def _format_mood(mood: dict | None) -> str:
    if not mood:
        return ""
    items = sorted(
        [(k, float(v)) for k, v in mood.items() if float(v) > 0.0],
        key=lambda kv: -kv[1],
    )
    if not items:
        return ""
    return ", ".join(f"{k}:{v:.1f}" for k, v in items)


def _format_state(user_state: dict | None) -> str:
    """Build compact one-line state projection identical to cosine_state baseline."""
    if not user_state:
        return ""
    parts: list[str] = []

    mood_str = _format_mood(user_state.get("mood_vector") or {})
    if mood_str:
        parts.append(f"mood: {mood_str}")

    bio_keys = ("hrv", "hrv_trend", "hr_trend", "sleep_quality", "sleep_hours",
                "stress_proxy", "time_of_day")
    for k in bio_keys:
        if user_state.get(k) is not None:
            label = "stress" if k == "stress_proxy" else ("time" if k == "time_of_day" else k)
            parts.append(f"{label}: {user_state[k]}")

    if user_state.get("recent_life_events_7d"):
        ev = ", ".join(str(x) for x in user_state["recent_life_events_7d"])
        parts.append(f"recent: {ev}")

    return "; ".join(parts)


def build_query_string(user_query: str, user_state: dict | None) -> str:
    """Concatenate query + ' | <state-string>' when state is present."""
    state_str = _format_state(user_state)
    if not state_str:
        return user_query
    return f"{user_query} | {state_str}"


# ─── Recall metric ────────────────────────────────────────────────────────────

def recall_at_3(retrieved: list[int], ideal: set[int]) -> float:
    if not ideal:
        return 0.0
    return len(set(retrieved[:3]) & ideal) / len(ideal)


# ─── Cross-encoder scoring ────────────────────────────────────────────────────

def load_cross_encoder(model_name: str):
    """Load cross-encoder model. Deferred import so CLI arg-parse works without GPU."""
    from sentence_transformers import CrossEncoder  # noqa: PLC0415
    print(f"[cross_encoder] loading {model_name} …", file=sys.stderr)
    t0 = time.time()
    ce = CrossEncoder(model_name, max_length=512)
    print(f"[cross_encoder] loaded in {time.time()-t0:.1f}s", file=sys.stderr)
    return ce


def score_pairs(ce, query_string: str, event_texts: list[str],
                batch_size: int = 64) -> list[float]:
    """Score (query_string, event_text) pairs. Returns float list same length as event_texts."""
    pairs = [(query_string, et) for et in event_texts]
    scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(s) for s in scores]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="State-conditioned cross-encoder baseline for bench v3")
    ap.add_argument("--corpus", type=Path, required=True,
                    help="bench/datasets/empathic-memory-corpus-v3.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Events to return per test (default 5)")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3",
                    help="HuggingFace cross-encoder model name "
                         "(default: BAAI/bge-reranker-v2-m3)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Pair scoring batch size (default 64)")
    args = ap.parse_args()

    # Load corpus
    data = json.loads(args.corpus.read_text(encoding="utf-8"))
    events: list[dict] = data["events"]
    tests: list[dict] = data["tests"]
    event_texts = [e["text"] for e in events]
    event_ids = [e["id"] for e in events]

    print(f"[cross_encoder] corpus: {len(events)} events, {len(tests)} tests",
          file=sys.stderr)

    ce = load_cross_encoder(args.model)

    per_test: list[dict] = []
    overall_list: list[float] = []
    per_type: dict[str, list[float]] = {}

    t_start = time.time()
    for ti, test in enumerate(tests, 1):
        ideal = set(test.get("ideal_top_3_event_ids") or [])
        ttype = test.get("test_type", "unknown")
        user_state = test.get("user_state") or test.get("state")

        query_str = build_query_string(test["user_query"], user_state)
        scores = score_pairs(ce, query_str, event_texts, batch_size=args.batch_size)

        # Sort by score descending
        ranked = sorted(zip(event_ids, scores), key=lambda x: -x[1])
        top_ids = [eid for eid, _ in ranked[: args.top_k]]

        r3 = recall_at_3(top_ids, ideal)
        overall_list.append(r3)
        per_type.setdefault(ttype, []).append(r3)

        per_test.append({
            "test_id": test.get("id"),
            "name": test.get("name"),
            "test_type": ttype,
            "user_query": test["user_query"],
            "state_projection": _format_state(user_state) or None,
            "query_string": query_str,
            "ideal_top_3": sorted(list(ideal)),
            "cross_encoder_top_5": top_ids,
            "recall_at_3": r3,
        })
        print(f"  [{ti:02d}/{len(tests)}] {test.get('name', '?')[:38]:.<40} "
              f"type={ttype:.<14} R@3={r3:.2f}", file=sys.stderr)

    runtime_s = time.time() - t_start
    overall_r = sum(overall_list) / len(overall_list) if overall_list else 0.0

    summary = {
        "n_events": len(events),
        "n_tests": len(tests),
        "overall_recall_at_3": round(overall_r, 6),
        "per_type_recall_at_3": {
            k: round(sum(v) / len(v), 6) for k, v in per_type.items()
        },
        "backend": {
            "system": "cross_encoder",
            "model": args.model,
            "state_projection": "query | mood: ...; hrv: ...; sleep: ...; "
                                "(same serialisation as cosine_state baseline)",
            "runtime_s": round(runtime_s, 1),
        },
        "per_test": per_test,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[summary] overall R@3 = {overall_r:.3f}", file=sys.stderr)
    for k, v_list in per_type.items():
        print(f"  {k:.<20} R@3 = {sum(v_list)/len(v_list):.3f}", file=sys.stderr)
    print(f"[runtime] {runtime_s:.1f}s for {len(tests)} tests × {len(events)} events "
          f"= {len(tests)*len(events)} pairs", file=sys.stderr)
    print(f"[save] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
