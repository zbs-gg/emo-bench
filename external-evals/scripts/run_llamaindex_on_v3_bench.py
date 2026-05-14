"""Run LlamaIndex on our empathic-memory-bench-v3 corpus.

Symmetric to run_mem0_on_v3_bench.py and run_graphiti_on_v3_bench.py.

Architecture:
  - Each event text is ingested as a TextNode with metadata={'event_id': int}
    into a VectorStoreIndex backed by SimpleVectorStore (in-memory, no infra).
  - At retrieval time, the VectorIndexRetriever returns NodeWithScore objects;
    we read event_id from each node's metadata and dedupe to top-K unique ids.
  - Backend: gpt-4o-mini (unused at retrieval, only models= matters) +
    text-embedding-3-small for embedding. ~$0.20 for 60 events + 35 queries.

No LLM extraction pass (unlike Mem0 which re-extracts facts via gpt-4o-mini).
LlamaIndex's VectorStoreIndex embeds the raw event text directly — cleanest
apples-to-apples embedding comparison.

Output schema is identical to Mem0/Graphiti adapters so downstream aggregators
can union all three.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


BACKEND_DEFAULTS = {
    "openai": {
        "llm_model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
    }
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run LlamaIndex VectorStoreIndex on the bench-v3 corpus."
    )
    ap.add_argument("--corpus", type=Path, required=True,
                    help="Path to empathic-memory-corpus-v3.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON result file")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of unique event_ids to return per query (default 5)")
    ap.add_argument("--backend", type=str, default="openai", choices=["openai"],
                    help="Backend: currently only 'openai' (gpt-4o-mini + text-embedding-3-small)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set. Run: set -a; source bench/.env; set +a")

    # Lazy imports after arg-check so --help works without heavy deps
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.schema import TextNode
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.vector_stores.simple import SimpleVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding

    cfg = BACKEND_DEFAULTS[args.backend]
    embed_model_name = cfg["embed_model"]
    llm_model_name = cfg["llm_model"]

    # Configure LlamaIndex global settings
    Settings.embed_model = OpenAIEmbedding(model=embed_model_name)
    Settings.llm = None  # retrieval-only — no LLM needed

    print(f"[llamaindex] backend={args.backend} embed={embed_model_name}", file=sys.stderr)

    data = json.loads(args.corpus.read_text())
    events: list[dict] = data["events"]
    tests: list[dict] = data["tests"]

    print(f"[ingest] {len(events)} events → embedding with {embed_model_name}", file=sys.stderr)
    t0 = time.time()

    # Build TextNodes — one per event, event_id in metadata
    nodes: list[TextNode] = []
    fail = 0
    for ev in events:
        text = ev.get("text", "")
        if not text:
            print(f"  [skip] event id={ev['id']} has no text", file=sys.stderr)
            fail += 1
            continue
        nodes.append(
            TextNode(
                text=text,
                metadata={"event_id": ev["id"]},
                # exclude event_id from embedding so the raw event text
                # is what gets embedded, not "event_id=N text=..."
                excluded_embed_metadata_keys=["event_id"],
                excluded_llm_metadata_keys=["event_id"],
            )
        )

    # Build the vector index — this embeds all nodes via one batched API call
    vs = SimpleVectorStore()
    sc = StorageContext.from_defaults(vector_store=vs)
    try:
        index = VectorStoreIndex(nodes, storage_context=sc, show_progress=True)
    except Exception as ex:
        sys.exit(f"ERROR during indexing: {ex}")

    ingest_time = time.time() - t0
    print(f"[ingest] done {len(nodes)-fail}/{len(events)} in {ingest_time:.1f}s (fail={fail})",
          file=sys.stderr)

    # Build retriever
    retriever = index.as_retriever(similarity_top_k=args.top_k * 3)

    print(f"[retrieve] {len(tests)} tests (top_k={args.top_k})", file=sys.stderr)
    t1 = time.time()

    results: list[dict] = []
    overall_recall = 0.0
    per_type_recall: dict[str, list[float]] = {}

    for ti, test in enumerate(tests, 1):
        query: str = test["user_query"]
        ideal: set[int] = set(test.get("ideal_top_3_event_ids", []))
        ttype: str = test.get("test_type", "unknown")

        try:
            raw_results = retriever.retrieve(query)
        except Exception as ex:
            print(f"  [search fail t={ti}] {ex}", file=sys.stderr)
            raw_results = []

        # Dedupe → top-K unique event_ids in score order
        retrieved_ids: list[int] = []
        seen: set[int] = set()
        for node_score in raw_results:
            eid = node_score.node.metadata.get("event_id")
            if eid is not None and eid not in seen:
                seen.add(eid)
                retrieved_ids.append(eid)
            if len(retrieved_ids) >= args.top_k:
                break

        retrieved_set = set(retrieved_ids[:3])
        recall = (len(retrieved_set & ideal) / len(ideal)) if ideal else 0.0
        overall_recall += recall
        per_type_recall.setdefault(ttype, []).append(recall)

        results.append({
            "test_id": test.get("id"),
            "name": test.get("name"),
            "test_type": ttype,
            "user_query": query,
            "ideal_top_3": list(ideal),
            "llamaindex_top_5": retrieved_ids[:5],
            "recall_at_3": recall,
        })
        print(
            f"  [{ti:>2}/{len(tests)}] {test.get('name','?'):.<40} "
            f"type={ttype:.<14} R@3={recall:.2f}  "
            f"retrieved={retrieved_ids[:5]}",
            file=sys.stderr,
        )

    retrieve_time = time.time() - t1
    overall_recall /= max(len(tests), 1)

    summary = {
        "n_events": len(events),
        "n_tests": len(tests),
        "overall_recall_at_3": round(overall_recall, 4),
        "per_type_recall_at_3": {
            k: round(sum(v) / len(v), 4) for k, v in per_type_recall.items()
        },
        "backend": {
            "llm_model": llm_model_name,
            "embed_model": embed_model_name,
            "vector_store": "SimpleVectorStore (in-memory)",
            "index": "VectorStoreIndex",
            "ingest_time_s": round(ingest_time, 1),
            "retrieve_time_s": round(retrieve_time, 1),
        },
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
