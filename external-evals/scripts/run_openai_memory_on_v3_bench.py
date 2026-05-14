"""Run OpenAI Vector Store + direct search on empathic-memory-bench-v3 corpus.

API surface used: OpenAI Vector Stores (client.vector_stores.*), SDK v2.32+.
This is the closest API-level analogue to "OpenAI Memory" at the time of
writing (May 2026). There is no cross-session or stateful memory primitive
in the OpenAI API beyond:
  - Assistants API (file_search tool) — wraps vector stores but requires
    an assistant + thread per query; adds latency, obscures ranking.
  - Responses API (store=True) — stores completions for retrieval, not
    semantic memory over user events.
  - ChatGPT memory — UI-only, no API surface.
  - beta.assistants.memory — does not exist in the SDK.

We use the DIRECT path: vector_stores.search() with per-file attributes
{"event_id": float(id)} so results map back to corpus event IDs without
any intermediate LLM or thread. This is deliberately a "best case for
OpenAI" baseline: clean embedding-based retrieval with no added extraction
overhead. If this doesn't beat random, it confirms our claim.

Architecture:
  Ingest:
    1. For each event, write text to a tmp .txt file named event_{id}.txt
    2. Upload via client.files.create(purpose="assistants")
    3. Attach to vector store via vector_stores.files.create_and_poll()
       with attributes={"event_id": float(id)}  (16-kv dict, float values)
    4. Batch-wait for all files to finish processing
  Retrieval:
    - client.vector_stores.search(vs_id, query=..., max_num_results=20)
    - Each result carries .attributes["event_id"] and .filename
    - Dedupe by event_id, take top-K unique event_ids

Cost estimate: ~60 file uploads (tiny text files) + 35 vector searches.
Text-embedding-3-large is used internally by OpenAI — we pay per token.
60 events × ~150 tokens avg + 35 queries × ~30 tokens ≈ ~10k tokens total.
At $0.13/1M tokens for text-embedding-3-large → <$0.01. Practically free.
File storage: free tier allows up to 1 GB. We clean up after.

Output schema matches mem0/langmem/graphiti adapters exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def make_event_text(ev: dict) -> str:
    """Bare event text — symmetric to other adapters' 'plain' variant."""
    return ev.get("text", "")


def cleanup_vector_store(client, vs_id: str, file_ids: list[str]) -> None:
    """Delete vector store and all uploaded files to keep account clean."""
    try:
        client.vector_stores.delete(vs_id)
    except Exception as e:
        print(f"  [cleanup] vector_store delete failed: {e}", file=sys.stderr)
    for fid in file_ids:
        try:
            client.files.delete(fid)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(
        description="Run OpenAI Vector Store search on bench v3"
    )
    ap.add_argument("--corpus", type=Path, required=True,
                    help="bench/datasets/empathic-memory-corpus-v3.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path")
    ap.add_argument("--top-k", type=int, default=5,
                    help="How many unique event_ids to return per query")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Skip deleting vector store and files after run (debug)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY must be set")

    import openai
    client = openai.OpenAI(api_key=api_key)

    data = json.loads(args.corpus.read_text())
    events = data["events"]
    tests = data["tests"]

    print(f"[openai-vs] SDK version: openai={openai.__version__}", file=sys.stderr)
    print(f"[openai-vs] corpus: {len(events)} events, {len(tests)} tests", file=sys.stderr)

    # ── Step 1: Create vector store ──────────────────────────────────────────
    run_name = f"bench_v3_{int(time.time())}"
    print(f"[openai-vs] creating vector store '{run_name}'", file=sys.stderr)
    vs = client.vector_stores.create(name=run_name)
    vs_id = vs.id
    print(f"[openai-vs] vector store id: {vs_id}", file=sys.stderr)

    uploaded_file_ids: list[str] = []
    ingest_t0 = time.time()
    ingest_fail = 0

    # ── Step 2: Upload each event as a text file ──────────────────────────────
    print(f"[ingest] uploading {len(events)} event files", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, ev in enumerate(events, 1):
            eid = ev["id"]
            text = make_event_text(ev)
            tmp_path = Path(tmpdir) / f"event_{eid}.txt"
            tmp_path.write_text(text, encoding="utf-8")
            try:
                # Upload to Files API
                with open(tmp_path, "rb") as fh:
                    file_obj = client.files.create(
                        file=(f"event_{eid}.txt", fh, "text/plain"),
                        purpose="assistants",
                    )
                uploaded_file_ids.append(file_obj.id)

                # Attach to vector store with event_id in attributes
                client.vector_stores.files.create_and_poll(
                    vector_store_id=vs_id,
                    file_id=file_obj.id,
                    attributes={"event_id": float(eid)},
                )
            except Exception as ex:
                ingest_fail += 1
                print(f"  [ingest fail id={eid}] {ex}", file=sys.stderr)

            if i % 10 == 0 or i == len(events):
                elapsed = time.time() - ingest_t0
                print(f"  ingest {i}/{len(events)} ({elapsed:.0f}s, fails={ingest_fail})",
                      file=sys.stderr)

    print(f"[ingest] done {len(events) - ingest_fail}/{len(events)} "
          f"in {time.time() - ingest_t0:.0f}s", file=sys.stderr)

    # ── Step 3: Retrieve ───────────────────────────────────────────────────────
    print(f"[retrieve] {len(tests)} tests", file=sys.stderr)
    results = []
    overall_recall = 0.0
    per_type_recall: dict[str, list[float]] = {}
    total_tokens_used = 0

    for ti, test in enumerate(tests, 1):
        query = test["user_query"]
        ideal = set(test.get("ideal_top_3_event_ids", []))
        ttype = test.get("test_type", "unknown")

        try:
            # Direct vector store search — no assistant or thread needed
            page = client.vector_stores.search(
                vs_id,
                query=query,
                max_num_results=min(args.top_k * 4, 50),
            )
            items = list(page)
        except Exception as ex:
            print(f"  [search fail t{ti}] {ex}", file=sys.stderr)
            items = []

        retrieved_ids: list[int] = []
        seen: set[int] = set()
        for it in items:
            # Primary: attributes["event_id"]
            eid = None
            if it.attributes and "event_id" in it.attributes:
                try:
                    eid = int(it.attributes["event_id"])
                except (ValueError, TypeError):
                    pass
            # Fallback: parse filename "event_{id}.txt"
            if eid is None and it.filename:
                fname = Path(it.filename).stem  # "event_42"
                if fname.startswith("event_"):
                    try:
                        eid = int(fname[len("event_"):])
                    except ValueError:
                        pass

            if eid is not None and eid not in seen:
                seen.add(eid)
                retrieved_ids.append(eid)
            if len(retrieved_ids) >= args.top_k:
                break

        retrieved_set = set(retrieved_ids[:3])
        recall = len(retrieved_set & ideal) / len(ideal) if ideal else 0.0
        overall_recall += recall
        per_type_recall.setdefault(ttype, []).append(recall)

        results.append({
            "test_id": test.get("id"),
            "name": test.get("name"),
            "test_type": ttype,
            "user_query": query,
            "ideal_top_3": sorted(ideal),
            "openai_memory_top_5": retrieved_ids[:5],
            "recall_at_3": recall,
        })
        print(f"  [{ti}/{len(tests)}] {test.get('name', '?'):.<40} type={ttype:.<14} "
              f"R@3={recall:.2f}", file=sys.stderr)

    overall_recall /= max(len(tests), 1)
    total_time = time.time() - ingest_t0

    # ── Step 4: Cleanup ────────────────────────────────────────────────────────
    if not args.no_cleanup:
        print(f"[cleanup] deleting vector store {vs_id} and {len(uploaded_file_ids)} files",
              file=sys.stderr)
        cleanup_vector_store(client, vs_id, uploaded_file_ids)
    else:
        print(f"[cleanup] skipped (--no-cleanup). vs_id={vs_id}", file=sys.stderr)

    # ── Step 5: Output ─────────────────────────────────────────────────────────
    summary = {
        "n_events": len(events),
        "n_tests": len(tests),
        "overall_recall_at_3": overall_recall,
        "per_type_recall_at_3": {k: sum(v) / len(v) for k, v in per_type_recall.items()},
        "backend": {
            "system": "openai-vector-store-file-search",
            "embed_model": "text-embedding-3-large (OpenAI default for vector stores)",
            "search_api": "client.vector_stores.search() direct (no assistant/thread)",
            "sdk_version": openai.__version__,
            "ingest_strategy": "one txt file per event, attributes={event_id: float}",
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
