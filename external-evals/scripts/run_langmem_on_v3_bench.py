"""Run LangMem (langchain-ai/langmem) on our empathic-memory-bench-v3 corpus.

Symmetric to run_mem0_on_v3_bench.py and run_graphiti_on_v3_bench.py.
Goal: defensible "Pulse beats LangMem" claim with shared backend.

Backend: OpenAI gpt-4o-mini + text-embedding-3-small (~$0.50 full run).
Storage: InMemoryStore with vector index (no Docker, no Postgres, no Neo4j).

Architecture:
  Ingest:
    - Each event text is stored raw (no LLM extraction) via InMemoryStore.put()
      with namespace ("bench_v3", "events") and key "event_{id}".
    - The value dict contains "text" and "event_id" so retrieval can map back.
  Retrieval:
    - InMemoryStore.search(namespace_prefix, query=..., limit=top_k) returns
      SearchItems ranked by embedding similarity.
    - We walk results, dedupe by event_id, collect top-K unique event_ids.

Output schema mirrors Mem0/Graphiti adapters so downstream aggregators can union.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ---- Backend selection ----
# Two modes:
#   "openai" — Real OpenAI gpt-4o-mini + text-embedding-3-small. ~$0.50 per
#              full 60-event/35-test run. Default, defensible comparison.
#   "local"  — Reserved for future LM Studio integration.
BACKEND = os.environ.get("BENCH_BACKEND", "openai")

if BACKEND == "openai":
    LLM_MODEL = os.environ.get("BENCH_LLM_MODEL", "gpt-4o-mini")
    EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "text-embedding-3-small")
    EMBED_DIMS = 1536  # text-embedding-3-small output dims
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: BENCH_BACKEND=openai requires OPENAI_API_KEY env var")
else:
    # local path — not fully wired, stub for future
    LLM_MODEL = os.environ.get("BENCH_LLM_MODEL", "bench-active")
    EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "nomic-embed")
    EMBED_DIMS = 768
    print(f"[langmem] WARNING: local backend not fully tested — use openai", file=sys.stderr)


def make_event_text(ev: dict) -> str:
    """Match Mem0/Graphiti adapters: bare event text only."""
    return ev.get("text", "")


def main():
    ap = argparse.ArgumentParser(
        description="Run LangMem (InMemoryStore + text-embedding-3-small) on bench v3"
    )
    ap.add_argument("--corpus", type=Path, required=True,
                    help="bench/datasets/empathic-memory-corpus-v3.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path")
    ap.add_argument("--top-k", type=int, default=5,
                    help="How many unique event_ids to return per query")
    ap.add_argument("--db-path", type=str, default=None,
                    help="Unused (InMemoryStore is ephemeral). Kept for CLI symmetry with Graphiti.")
    ap.add_argument("--namespace", type=str, default="bench_v3",
                    help="LangMem namespace prefix for this run")
    ap.add_argument("--backend", type=str, default="openai", choices=["openai", "local"],
                    help="Backend to use (default: openai)")
    args = ap.parse_args()

    # Allow --backend CLI arg to override env
    if args.backend != BACKEND:
        os.environ["BENCH_BACKEND"] = args.backend
        if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
            sys.exit("ERROR: --backend openai requires OPENAI_API_KEY env var")

    # Lazy imports after env check
    from langchain_openai import OpenAIEmbeddings
    from langgraph.store.memory import InMemoryStore

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    store = InMemoryStore(
        index={
            "dims": EMBED_DIMS,
            "embed": embeddings,
            "fields": ["text"],  # field to embed for vector search
        }
    )

    print(f"[langmem] backend: LLM={LLM_MODEL} embed={EMBED_MODEL}", file=sys.stderr)
    print(f"[langmem] store: InMemoryStore (ephemeral, vector-indexed)", file=sys.stderr)
    print(f"[langmem] namespace: {args.namespace}", file=sys.stderr)

    data = json.loads(args.corpus.read_text())
    events = data["events"]
    tests = data["tests"]

    namespace = (args.namespace, "events")

    print(f"[ingest] {len(events)} events", file=sys.stderr)
    t0 = time.time()
    fail = 0
    for i, ev in enumerate(events, 1):
        eid = ev["id"]
        key = f"event_{eid}"
        text = make_event_text(ev)
        value = {
            "text": text,
            "event_id": eid,
        }
        try:
            store.put(namespace, key, value)
        except Exception as ex:
            fail += 1
            print(f"  [put fail id={eid}] {ex}", file=sys.stderr)
        if i % 10 == 0:
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
            # search returns SearchItem list, sorted by score desc
            items = store.search(
                namespace,
                query=query,
                limit=args.top_k * 3,  # over-fetch to allow dedup
            )
        except Exception as ex:
            print(f"  [search fail t{ti}] {ex}", file=sys.stderr)
            items = []

        retrieved_ids: list[int] = []
        seen: set[int] = set()
        for item in items:
            eid = item.value.get("event_id")
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
            "ideal_top_3": sorted(ideal),
            "langmem_top_5": retrieved_ids[:5],
            "recall_at_3": recall,
        })
        print(f"  [{ti}/{len(tests)}] {test.get('name', '?'):.<40} type={ttype:.<14} "
              f"R@3={recall:.2f}", file=sys.stderr)

    overall_recall /= max(len(tests), 1)
    total_time = time.time() - t0

    summary = {
        "n_events": len(events),
        "n_tests": len(tests),
        "overall_recall_at_3": overall_recall,
        "per_type_recall_at_3": {k: sum(v) / len(v) for k, v in per_type_recall.items()},
        "backend": {
            "llm_model": LLM_MODEL,
            "embed_model": EMBED_MODEL,
            "store": "InMemoryStore",
            "backend": BACKEND,
        },
        "runtime_seconds": round(total_time, 1),
        "per_test": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[summary] overall R@3 = {overall_recall:.3f}", file=sys.stderr)
    for k, v in summary["per_type_recall_at_3"].items():
        print(f"  {k:.<20} R@3 = {v:.3f}", file=sys.stderr)
    print(f"[runtime] {total_time:.0f}s", file=sys.stderr)
    print(f"[save] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
