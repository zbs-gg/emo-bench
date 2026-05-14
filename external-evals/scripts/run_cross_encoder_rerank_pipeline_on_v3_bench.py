"""Standard IR pipeline: bi-encoder retrieval → cross-encoder rerank.

Addresses reviewer delta's concern that the prior cross-encoder baseline
(bge-reranker-v2-m3) was run in a non-standard mode — scoring all 60 events
directly rather than reranking a top-K candidate pool from a bi-encoder first.

Standard pipeline:
  1. Bi-encoder (text-embedding-3-small via OpenAI API): embed query+state and
     all 60 events. Cosine similarity → top-K candidates (default K=20).
  2. Cross-encoder (BAAI/bge-reranker-v2-m3): score each top-K (query+state,
     event_text) pair. Sort descending. Take top-3 for R@3 evaluation.

Output JSON schema matches the existing cross-encoder JSON, with added
`pipeline_top_k_bienc` key in the backend metadata block.

Usage:
  python3 external-evals/scripts/run_cross_encoder_rerank_pipeline_on_v3_bench.py \\
    --corpus datasets/empathic-memory-corpus-v3.json \\
    --out external-evals/results/cross-encoder-rerank-pipeline-v3-bge-reranker-<stamp>.json \\
    --top-k-bienc 20 \\
    --top-k 5

Venv: bench/.venv-cross-encoder (sentence-transformers + torch already installed).
OPENAI_API_KEY must be set (or loaded from bench/.env) for the bi-encoder step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


# ─── Helpers: state serialisation ─────────────────────────────────────────────
# Identical to run_cross_encoder_on_v3_bench.py so comparisons are apples-to-apples.

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
    """Compact one-line state projection identical to cosine_state baseline."""
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
    state_str = _format_state(user_state)
    if not state_str:
        return user_query
    return f"{user_query} | {state_str}"


# ─── Metric ───────────────────────────────────────────────────────────────────

def recall_at_3(retrieved: list[int], ideal: set[int]) -> float:
    if not ideal:
        return 0.0
    return len(set(retrieved[:3]) & ideal) / len(ideal)


# ─── Bi-encoder (OpenAI text-embedding-3-small) ───────────────────────────────

def _openai_embed(texts: list[str], model: str, api_key: str,
                  batch: int = 64) -> np.ndarray:
    """Embed texts via OpenAI Embeddings API. Returns L2-normalised float32 (N, D)."""
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i: i + batch]
        body = json.dumps({"input": chunk, "model": model}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        for item in d["data"]:
            vecs.append(item["embedding"])
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cosine_top_k(query_vec: np.ndarray, event_vecs: np.ndarray, k: int) -> list[int]:
    """Return indices of top-K events by cosine similarity (both vecs L2-normalised)."""
    sims = event_vecs @ query_vec  # shape (N,) — dot on unit vecs = cosine
    # argsort ascending, take last k reversed
    idx = np.argsort(sims)[-k:][::-1]
    return idx.tolist()


# ─── Cross-encoder ────────────────────────────────────────────────────────────

def load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder  # noqa: PLC0415
    print(f"[pipeline] loading cross-encoder {model_name} …", file=sys.stderr)
    t0 = time.time()
    ce = CrossEncoder(model_name, max_length=512)
    print(f"[pipeline] cross-encoder loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    return ce


def ce_score_candidates(ce, query_string: str, candidate_texts: list[str],
                         batch_size: int = 64) -> list[float]:
    """Score (query_string, candidate_text) pairs via cross-encoder."""
    pairs = [(query_string, ct) for ct in candidate_texts]
    scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(s) for s in scores]


# ─── .env loader ──────────────────────────────────────────────────────────────

def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — sets os.environ for lines matching KEY=VALUE."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bi-encoder retrieval → cross-encoder rerank pipeline for bench v3")
    ap.add_argument("--corpus", type=Path, required=True,
                    help="bench/datasets/empathic-memory-corpus-v3.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path")
    ap.add_argument("--top-k-bienc", type=int, default=20,
                    help="Bi-encoder candidate pool size (default 20)")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Final events to return per test (default 5)")
    ap.add_argument("--bienc-model", default="text-embedding-3-small",
                    help="OpenAI embedding model for bi-encoder stage "
                         "(default: text-embedding-3-small)")
    ap.add_argument("--ce-model", default="BAAI/bge-reranker-v2-m3",
                    help="HuggingFace cross-encoder model (default: BAAI/bge-reranker-v2-m3)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Cross-encoder pair scoring batch size (default 64)")
    ap.add_argument("--env", type=Path,
                    default=Path(__file__).parent.parent.parent / ".env",
                    help="Path to .env file (default: bench/.env)")
    args = ap.parse_args()

    # Load .env so OPENAI_API_KEY is available when running from any cwd
    _load_dotenv(args.env)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[pipeline] ERROR: OPENAI_API_KEY not set — needed for bi-encoder step",
              file=sys.stderr)
        return 1

    # ── Load corpus ──────────────────────────────────────────────────────────
    data = json.loads(args.corpus.read_text(encoding="utf-8"))
    events: list[dict] = data["events"]
    tests: list[dict] = data["tests"]
    event_texts = [e["text"] for e in events]
    event_ids = [e["id"] for e in events]
    n_ev = len(events)
    n_tests = len(tests)
    print(f"[pipeline] corpus: {n_ev} events, {n_tests} tests", file=sys.stderr)

    # ── Step 1: embed all events via bi-encoder ──────────────────────────────
    print(f"[pipeline] bi-encoder: embedding {n_ev} events via {args.bienc_model} …",
          file=sys.stderr)
    t_emb0 = time.time()
    event_vecs = _openai_embed(event_texts, model=args.bienc_model, api_key=api_key)
    t_emb_events = time.time() - t_emb0
    print(f"[pipeline] event embeddings done in {t_emb_events:.1f}s "
          f"(shape {event_vecs.shape})", file=sys.stderr)

    # ── Step 2: load cross-encoder ───────────────────────────────────────────
    ce = load_cross_encoder(args.ce_model)

    # ── Step 3: per-test pipeline ────────────────────────────────────────────
    per_test: list[dict] = []
    overall_list: list[float] = []
    per_type: dict[str, list[float]] = {}

    t_start = time.time()
    for ti, test in enumerate(tests, 1):
        ideal = set(test.get("ideal_top_3_event_ids") or [])
        ttype = test.get("test_type", "unknown")
        user_state = test.get("user_state") or test.get("state")

        query_str = build_query_string(test["user_query"], user_state)

        # ── Bi-encoder: embed query, get top-K candidates ────────────────────
        q_vec = _openai_embed([query_str], model=args.bienc_model, api_key=api_key)[0]
        # top_k_bienc capped at corpus size
        k_bienc = min(args.top_k_bienc, n_ev)
        cand_indices = cosine_top_k(q_vec, event_vecs, k=k_bienc)
        cand_ids = [event_ids[i] for i in cand_indices]
        cand_texts = [event_texts[i] for i in cand_indices]

        # ── Cross-encoder: score candidates, rerank ──────────────────────────
        ce_scores = ce_score_candidates(ce, query_str, cand_texts,
                                        batch_size=args.batch_size)
        reranked = sorted(zip(cand_ids, ce_scores), key=lambda x: -x[1])
        top_ids = [eid for eid, _ in reranked[: args.top_k]]

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
            "biencoder_top_k_candidates": cand_ids,
            "cross_encoder_top_k": top_ids,
            "recall_at_3": r3,
        })
        print(f"  [{ti:02d}/{n_tests}] {test.get('name', '?')[:38]:.<40} "
              f"type={ttype:.<14} R@3={r3:.2f}", file=sys.stderr)

    runtime_s = time.time() - t_start
    overall_r = sum(overall_list) / len(overall_list) if overall_list else 0.0

    summary = {
        "n_events": n_ev,
        "n_tests": n_tests,
        "overall_recall_at_3": round(overall_r, 6),
        "per_type_recall_at_3": {
            k: round(sum(v) / len(v), 6) for k, v in per_type.items()
        },
        "backend": {
            "system": "cross_encoder_rerank_pipeline",
            "bienc_model": args.bienc_model,
            "ce_model": args.ce_model,
            "pipeline_top_k_bienc": args.top_k_bienc,
            "pipeline_top_k_final": args.top_k,
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
        print(f"  {k:.<20} R@3 = {sum(v_list) / len(v_list):.3f}", file=sys.stderr)
    print(f"[runtime] {runtime_s:.1f}s for {n_tests} tests × {k_bienc} candidates "
          f"(bi-encoder pool K={args.top_k_bienc} of {n_ev} events)", file=sys.stderr)
    print(f"[save] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
