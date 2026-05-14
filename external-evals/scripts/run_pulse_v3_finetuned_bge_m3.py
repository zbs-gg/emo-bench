"""C.4.4 — Zero-shot Pulse 35-probe evaluation with fine-tuned bge-m3.

CRITICAL: this is a STRICT zero-shot holdout. The fine-tuned bge-m3 LoRA adapter
was trained ONLY on public EmpatheticDialogues + ESConv triplets (see
gen_finetune_triplets.py). The Pulse 35-probe corpus is NEVER seen during
training.

Per Gemini 3.1 Pro's recommendation: this design neutralizes the 5/5 unanimous
"circular evaluation" peer-review concern structurally — model has zero
exposure to author data in training.

Output: external-evals/results/pulse_v3-finetuned-bge-m3-zero-shot-{ts}.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "finetune-adapters" / "bge-m3-empathic-2026-05"


def recall_at_3(retrieved, ideal):
    if not ideal: return 0.0
    return len(set(retrieved[:3]) & set(ideal)) / len(ideal)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=ADAPTER_PATH,
                    help="Path to fine-tuned bge-m3 LoRA adapter (must exist)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    if not args.adapter.exists():
        print(f"ERROR: adapter not found at {args.adapter}", file=sys.stderr)
        return 1

    # Override the embedding provider to our fine-tuned model via monkey-patch.
    # retrieval_v3 calls embed_cohere_or_alt which dispatches on
    # EMBEDDING_PROVIDER env. We add a 'finetuned-bge-m3' provider here.
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[finetuned-eval] loading fine-tuned bge-m3 from {args.adapter}", file=sys.stderr)
    t0 = time.time()
    model = SentenceTransformer(str(args.adapter), device=device)
    print(f"[finetuned-eval] loaded in {time.time()-t0:.1f}s on {device}", file=sys.stderr)

    def embed_batch(texts, input_type):
        # bge-m3 uses E5-style instruction prefixes, but for LoRA-tuned model
        # we encode plain text. retrieval_v3 passes "search_document" or
        # "search_query" — we ignore the input_type for now.
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    # Patch the embedding_provider module
    import embedding_provider
    embedding_provider.embed_texts = embed_batch

    # Now load retrieval_v3 with patched embedder
    from retrieval_v3 import RetrievalV3, UserState

    data = json.loads(args.corpus.read_text())
    events = data["events"]
    tests = data["tests"]

    print(f"[finetuned-eval] embedding {len(events)} events...", file=sys.stderr)
    t1 = time.time()
    engine = RetrievalV3(events, use_llm_query_emo=False)
    print(f"[finetuned-eval] events embedded in {time.time()-t1:.1f}s", file=sys.stderr)

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
            stress_proxy=s.get("stress_proxy"),
            recent_life_events_7d=s.get("recent_life_events_7d") or [],
            time_of_day=s.get("time_of_day"),
            snapshot_days_ago=s.get("snapshot_days_ago"),
        )

    per_test = []
    overall, per_type = [], {}
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
            "ideal_top_3": sorted(list(ideal)),
            "retrieved_top_5": retrieved[:5],
            "recall_at_3": r3,
        })
        print(f"  [{ti}/{len(tests)}] {t.get('name','?'):.<40} type={ttype:.<14} R@3={r3:.2f}", file=sys.stderr)

    overall_r = sum(overall)/len(overall) if overall else 0.0
    non_chain = [r for r, t in zip(overall, tests) if t.get("test_type") != "chain"]
    non_chain_r = sum(non_chain)/len(non_chain) if non_chain else 0.0

    summary = {
        "n_events": len(events),
        "n_tests": len(tests),
        "overall_recall_at_3": overall_r,
        "non_chain_recall_at_3": non_chain_r,
        "per_type_recall_at_3": {k: sum(v)/len(v) for k, v in per_type.items()},
        "backend": {
            "system": "pulse_v3",
            "embed_model": "bge-m3 + LoRA adapter (empathic-2026-05)",
            "adapter_path": str(args.adapter),
            "training_data": "public EmpatheticDialogues + ESConv triplets only — Pulse corpus NEVER seen during training",
            "note": "STRICT zero-shot holdout per Gemini 3.1 Pro recommendation",
        },
        "per_test": per_test,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[summary] overall R@3 = {overall_r:.4f}", file=sys.stderr)
    print(f"[summary] non-chain R@3 = {non_chain_r:.4f}", file=sys.stderr)
    for k, v in summary["per_type_recall_at_3"].items():
        print(f"  {k:.<20} R@3 = {v:.3f}", file=sys.stderr)
    print(f"[save] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
