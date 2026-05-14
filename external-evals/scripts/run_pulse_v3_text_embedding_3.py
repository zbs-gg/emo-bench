"""Run Pulse v3 retrieval with text-embedding-3-{small,large} backbone for
backbone-matched comparison against memory-system baselines (Mem0, Graphiti,
LangMem, LlamaIndex, OpenAI Memory) which all use OpenAI embeddings.

Output JSON matches the Mem0/Graphiti adapter schema so the aggregator can
union the new row into leaderboard-v3.{md,csv}."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


def recall_at_3(retrieved: list[int], ideal: set[int]) -> float:
    if not ideal:
        return 0.0
    return len(set(retrieved[:3]) & ideal) / len(ideal)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True,
                    help="bench/datasets/empathic-memory-corpus-v3.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="text-embedding-3-small",
                    choices=["text-embedding-3-small", "text-embedding-3-large"])
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    os.environ["EMBEDDING_PROVIDER"] = args.model

    from retrieval_v3 import RetrievalV3, UserState  # noqa: E402

    data = json.loads(args.corpus.read_text())
    events = data["events"]
    tests = data["tests"]

    print(f"[pulse_v3-{args.model}] embedding {len(events)} events…", file=sys.stderr)
    t0 = time.time()
    engine = RetrievalV3(events, use_llm_query_emo=False)
    print(f"[pulse_v3-{args.model}] events embedded in {time.time()-t0:.1f}s",
          file=sys.stderr)

    def state_from_test(t):
        s = t.get("user_state") or t.get("state") or {}
        if not s:
            return None
        return UserState(
            mood_vector=s.get("mood_vector") or {},
            sleep_quality=s.get("sleep_quality"),
            sleep_hours=s.get("sleep_hours"),
            hrv=s.get("hrv"),
            hr_trend=s.get("hr_trend"),
            hrv_trend=s.get("hrv_trend"),
            stress_proxy=s.get("stress_proxy"),
            recent_life_events_7d=s.get("recent_life_events_7d") or [],
            time_of_day=s.get("time_of_day"),
            snapshot_days_ago=s.get("snapshot_days_ago"),
        )

    per_test = []
    overall, per_type = [], {}
    print(f"[pulse_v3-{args.model}] {len(tests)} probes", file=sys.stderr)
    for ti, t in enumerate(tests, 1):
        ideal = set(t.get("ideal_top_3_event_ids") or [])
        ttype = t.get("test_type", "unknown")
        is_chain = ttype == "chain"
        ids_with = engine.retrieve(
            t["user_query"],
            user_state=state_from_test(t),
            top_k=args.top_k,
            expand_chain=is_chain,
            return_scores=True,
        )
        retrieved = [int(eid) for eid, _ in ids_with]
        r3 = recall_at_3(retrieved, ideal)
        overall.append(r3)
        per_type.setdefault(ttype, []).append(r3)
        per_test.append({
            "test_id": t.get("id"),
            "name": t.get("name"),
            "test_type": ttype,
            "user_query": t["user_query"],
            "ideal_top_3": sorted(list(ideal)),
            "pulse_v3_top_5": retrieved[:5],
            "recall_at_3": r3,
        })
        print(f"  [{ti}/{len(tests)}] {t.get('name','?'):.<40} type={ttype:.<14} "
              f"R@3={r3:.2f}", file=sys.stderr)

    overall_r = sum(overall)/len(overall) if overall else 0.0
    summary = {
        "n_events": len(events),
        "n_tests": len(tests),
        "overall_recall_at_3": overall_r,
        "per_type_recall_at_3": {k: sum(v)/len(v) for k, v in per_type.items()},
        "backend": {
            "system": "pulse_v3",
            "embed_model": args.model,
            "endpoint": "https://api.openai.com/v1/embeddings",
            "note": "backbone-matched run for fair comparison vs memory-system "
                    "baselines that also use OpenAI text-embedding-3",
        },
        "per_test": per_test,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[summary] overall R@3 = {overall_r:.3f}", file=sys.stderr)
    for k, v in summary["per_type_recall_at_3"].items():
        print(f"  {k:.<20} R@3 = {v:.3f}", file=sys.stderr)
    print(f"[save] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
