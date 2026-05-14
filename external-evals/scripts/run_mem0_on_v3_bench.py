"""Run Mem0 on our empathic-memory-bench-v3 corpus.

Symmetric to run_mem0_locomo / run_mem0_lme — but on our home-court bench.
Goal: prove Mem0 cannot retrieve stateful/chain/multi-signal cases (no APIs
for emotion_tags / user_state / predecessor_ids).

We only measure retrieval quality (Recall@3 vs ideal_top_3_event_ids) so we
don't need API judges. Output: per-test mem0 retrievals + retrieval metrics.

Each event is added with metadata={"event_id": int}. Mem0.search returns
fact-fragments with metadata; we map fragments → event_ids and pick top-K
unique event_ids.

Per-test: per-axis judgement uses ideal_top_3_event_ids field as ground truth.
- Recall@3 = |retrieved ∩ ideal| / |ideal|
- IoU@3   = |retrieved ∩ ideal| / |retrieved ∪ ideal|

These are retrieval-quality proxies. Empathic axes from rubric (rel/spec/act/
chain/multi/stateful) require LLM judges — skipped here for the headline run.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path


def make_event_text(ev: dict, variant: str = "plain") -> str:
    """Compose event text. Variants:
      plain: bare event text (Mem0 default extraction, no emotion hints)
      custom_instructions: prepend [EMOTIONS: ...] tags so the emotion-aware
        extraction prompt has explicit signal to work with.
    """
    if variant == "plain":
        return ev.get("text", "")
    # custom_instructions: emotion-aware ingest
    parts = []
    if ev.get("emotion_tags"):
        top = sorted(ev["emotion_tags"].items(), key=lambda kv: -kv[1])[:3]
        nonzero = [(k, v) for k, v in top if v > 0.2]
        if nonzero:
            parts.append("[EMOTIONS: " + ", ".join(f"{k}={v:.2f}" for k, v in nonzero) + "]")
    if ev.get("sentiment_label"):
        parts.append(f"[label: {ev['sentiment_label']}]")
    if ev.get("days_ago") is not None:
        parts.append(f"[days_ago: {ev['days_ago']}]")
    parts.append(ev.get("text", ""))
    return "\n".join(parts)


# Custom instructions for emotion-aware extraction. Mem0 v2.0 injects this
# verbatim as a "highest priority" section in the extraction prompt
# (configs/prompts.py:546, 1044).
EMOTION_AWARE_INSTRUCTIONS = """
For every memory you extract:
1. Preserve the fact (who/what/when/where/why).
2. Identify and INCLUDE the emotional tone(s) of the event from the
   Plutchik-10 vocabulary: joy, trust, fear, surprise, sadness, disgust,
   anger, anticipation, shame, guilt. The user's events are pre-tagged
   with [EMOTIONS: ...] markers — propagate those tags into the memory.
3. When the event is emotionally charged (any tag &gt;= 0.5), explicitly
   note the dominant emotion in the memory text, e.g. "User expressed
   shame about X" or "User felt joyful anticipation about Y".
4. Do not strip or paraphrase away the emotion words — they are
   load-bearing for retrieval.
""".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True,
                    help="bench/datasets/empathic-memory-corpus-v3.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5,
                    help="How many unique event_ids to return per query")
    ap.add_argument("--qdrant-path", type=str, default="/tmp/qdrant_bench_v3")
    ap.add_argument("--variant", type=str, default="plain",
                    choices=["plain", "custom_instructions"],
                    help="plain = vanilla Mem0; custom_instructions = emotion-aware extraction")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: set OPENAI_API_KEY (Mem0 fact extraction uses gpt-4o-mini)")

    from mem0 import Memory
    cfg = {"vector_store": {"provider": "qdrant",
                            "config": {"path": args.qdrant_path,
                                       "collection_name": f"bench_v3_mem0_{args.variant}"}}}
    if args.variant == "custom_instructions":
        cfg["custom_instructions"] = EMOTION_AWARE_INSTRUCTIONS
    memory = Memory.from_config(cfg)
    print(f"[mem0 variant={args.variant}] initialized (storage={args.qdrant_path})",
          file=sys.stderr)

    data = json.loads(args.corpus.read_text())
    events = data["events"]
    tests = data["tests"]

    user_id = "bench_v3"
    # Wipe so this is a fresh ingest
    try:
        memory.delete_all(user_id=user_id)
    except Exception:
        pass

    print(f"[ingest] {len(events)} events", file=sys.stderr)
    t0 = time.time()
    fail = 0
    for i, ev in enumerate(events, 1):
        text = make_event_text(ev, variant=args.variant)
        # Pass emotion_tags as metadata so it's filterable on search side too
        meta = {"event_id": ev["id"]}
        if args.variant == "custom_instructions" and ev.get("emotion_tags"):
            top_emo = sorted(ev["emotion_tags"].items(), key=lambda kv: -kv[1])
            if top_emo and top_emo[0][1] > 0.3:
                meta["dominant_emotion"] = top_emo[0][0]
                meta["emotion_intensity"] = float(top_emo[0][1])
        try:
            memory.add(text, user_id=user_id, metadata=meta)
        except Exception as ex:
            fail += 1
            print(f"  [add fail id={ev['id']}] {ex}", file=sys.stderr)
        if i % 5 == 0:
            print(f"  ingest {i}/{len(events)} ({time.time()-t0:.0f}s, fails={fail})",
                  file=sys.stderr)
    print(f"[ingest] done {len(events)-fail}/{len(events)} in {time.time()-t0:.0f}s",
          file=sys.stderr)

    # Per-test retrieval
    print(f"[retrieve] {len(tests)} tests", file=sys.stderr)
    results = []
    overall_recall = 0.0
    per_type_recall: dict[str, list[float]] = {}
    for ti, test in enumerate(tests, 1):
        query = test["user_query"]
        ideal = set(test.get("ideal_top_3_event_ids", []))
        ttype = test.get("test_type", "unknown")
        try:
            r = memory.search(query, filters={"user_id": user_id}, limit=args.top_k * 4)
            items = r.get("results", []) if isinstance(r, dict) else []
        except Exception as ex:
            print(f"  [search fail t{ti}] {ex}", file=sys.stderr)
            items = []

        retrieved_ids = []
        seen = set()
        for it in items:
            md = it.get("metadata") or {}
            eid = md.get("event_id")
            if eid is not None and eid not in seen:
                seen.add(eid)
                retrieved_ids.append(eid)
            if len(retrieved_ids) >= args.top_k:
                break

        retrieved_set = set(retrieved_ids[:3])
        if ideal:
            recall = len(retrieved_set & ideal) / len(ideal)
        else:
            recall = 0.0
        overall_recall += recall
        per_type_recall.setdefault(ttype, []).append(recall)

        results.append({
            "test_id": test.get("id"),
            "name": test.get("name"),
            "test_type": ttype,
            "user_query": query,
            "ideal_top_3": list(ideal),
            "mem0_top_5": retrieved_ids[:5],
            "recall_at_3": recall,
        })
        print(f"  [{ti}/{len(tests)}] {test.get('name','?'):.<40} type={ttype:.<14} "
              f"R@3={recall:.2f}", file=sys.stderr)

    overall_recall /= max(len(tests), 1)
    summary = {
        "n_events": len(events),
        "n_tests": len(tests),
        "overall_recall_at_3": overall_recall,
        "per_type_recall_at_3": {k: sum(v) / len(v) for k, v in per_type_recall.items()},
        "per_test": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[summary] overall R@3 = {overall_recall:.3f}", file=sys.stderr)
    for k, v in summary["per_type_recall_at_3"].items():
        print(f"  {k:.<20} R@3 = {v:.3f}", file=sys.stderr)
    print(f"[save] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
