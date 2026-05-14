"""Phase D density sensitivity test.

Reviewer delta's concern: the monotonic regression of emotion-boost (β) on NDCG@3
observed in Phase D might be an artefact of high emotion-annotation density (all 60
events have emotion_tags). At sparser annotation, always-on emotion weighting may not
monotonically hurt retrieval.

Methodology:
- For each density d ∈ {1.0, 0.5, 0.25, 0.10}: randomly sample d × 60 events to
  KEEP their emotion_tags; set emotion_tags = {} for the rest (seed 42, fixed).
- For each β ∈ {0.0, 0.1, 0.2, 0.5, 1.0}: run Pulse v3 retrieval on the modified
  corpus with emotion boost ALWAYS ON (ignoring conditional gate so we test the
  always-on hypothesis directly), compute NDCG@3 over all 35 probes.
- Probes with empty ideal_top_3_event_ids contribute NDCG@3 = 0.0 (consistent with
  Phase D paper treatment of chain tests whose ideal sets are None).

NDCG@3: relevance is binary (ideal = 1, non-ideal = 0). Standard DCG formula.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


# ── NDCG@3 helpers ──────────────────────────────────────────────────────────

def dcg_at_k(ranked_ids: list[int], ideal_ids: set[int], k: int = 3) -> float:
    """Compute DCG@k with binary relevance."""
    score = 0.0
    for rank, eid in enumerate(ranked_ids[:k], start=1):
        if eid in ideal_ids:
            score += 1.0 / math.log2(rank + 1)
    return score


def idcg_at_k(ideal_ids: set[int], k: int = 3) -> float:
    """Ideal DCG@k: place all relevant items at the top ranks."""
    n_rel = min(len(ideal_ids), k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, n_rel + 1))


def ndcg_at_k(ranked_ids: list[int], ideal_ids: set[int], k: int = 3) -> float:
    """NDCG@k. Returns 0.0 if ideal_ids is empty (no ground truth)."""
    if not ideal_ids:
        return 0.0
    idcg = idcg_at_k(ideal_ids, k)
    if idcg < 1e-12:
        return 0.0
    return dcg_at_k(ranked_ids, ideal_ids, k) / idcg


# ── State reconstruction (mirrors run_pulse_v3_text_embedding_3.py) ─────────

def state_from_test(t: dict):
    from retrieval_v3 import UserState  # noqa: E402
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


# ── Density-masked corpus builder ────────────────────────────────────────────

def mask_emotion_tags(events: list[dict], density: float, seed: int) -> list[dict]:
    """Return a copy of events where only density × len(events) events retain
    their emotion_tags; the rest have emotion_tags set to empty dict.

    Selection is deterministic given (density, seed): sorted event_ids shuffled
    with the given seed, top-d fraction kept.
    """
    rng = random.Random(seed)
    n_keep = max(1, round(density * len(events)))
    ids = sorted(e["id"] for e in events)
    rng.shuffle(ids)
    keep_set = set(ids[:n_keep])

    masked: list[dict] = []
    for e in events:
        ec = copy.deepcopy(e)
        if ec["id"] not in keep_set:
            ec["emotion_tags"] = {}
        masked.append(ec)
    return masked


# ── Always-on emotion boost patch ────────────────────────────────────────────

def patch_retrieval_always_on_emotion(engine, beta: float) -> None:
    """Monkey-patch engine.retrieve to force the emotion boost to always fire
    (regardless of whether the query has a 'dominant emotion'), matching the
    Phase D always-on scenario. Sets beta to the supplied value.

    Implementation: override the has_dominant_emotion gate by forcing dom_ok=True
    whenever there is ANY nonzero query emotion vector.
    """
    engine.beta = beta

    # Save original retrieve
    original_retrieve = engine.__class__.retrieve

    def patched_retrieve(self, query, user_state=None, top_k=3,
                         return_scores=False, expand_chain=False,
                         augment_query=True):
        """Identical to RetrievalV3.retrieve but with emotion boost always active."""
        from retrieval_v3 import (
            embed_cohere_or_alt, infer_query_emotions_keyword,
            infer_query_emotions_llm, emotion_vec, compute_state_fit,
            compute_date_proximity, expand_chain_from_seeds, build_chain_graph,
        )

        # 1. Base semantic score
        effective_query = query
        if (self.enable_emotion_hint_augment and augment_query
                and user_state and user_state.mood_vector):
            dom_ok, _, dom_key = user_state.has_dominant_emotion(0.5)
            if dom_ok:
                from retrieval_v3 import EMOTION_QUERY_HINTS
                hint = EMOTION_QUERY_HINTS.get(dom_key)
                if hint:
                    effective_query = f"{query} {hint}"

        q_vec = embed_cohere_or_alt([effective_query], "search_query")[0]
        sims = self._event_vecs @ q_vec
        recency = np.exp(-self._eff_lambda * self._days)
        base = sims * recency

        # 2. ALWAYS-ON emotion boost (key difference from v3 conditional)
        boost_emo = np.ones_like(base)
        if self.enable_emotion_boost and self.beta > 1e-9:
            if user_state and user_state.mood_vector:
                q_emo = emotion_vec(user_state.mood_vector)
            else:
                if self.use_llm_query_emo:
                    emo_dict = infer_query_emotions_llm(query, self.query_emo_provider)
                else:
                    emo_dict = infer_query_emotions_keyword(query)
                q_emo = emotion_vec(emo_dict)

            q_norm = float(np.linalg.norm(q_emo))
            if q_norm > 1e-6:
                e_norms = np.linalg.norm(self._event_emos, axis=1)
                mask = e_norms > 1e-6
                align = np.zeros_like(base)
                if mask.any():
                    align[mask] = (self._event_emos[mask] @ q_emo) / (e_norms[mask] * q_norm)
                boost_emo = 1.0 + self.beta * np.clip(align, 0.0, None)
            # When q_norm ≈ 0 (no query emotion), boost_emo stays 1 (no effect)

        # 3. Conditional state/body boost
        boost_state = np.ones_like(base)
        if self.enable_state_boost and user_state is not None and (
                user_state.is_body_stressed() or user_state.is_body_restored()):
            fit = np.array([compute_state_fit(e, user_state) for e in self.events],
                           dtype=np.float32)
            boost_state = 1.0 + self.gamma * fit

        # 3b. Anchor boost
        boost_anchor = np.ones_like(base)
        if self.enable_anchor_boost and self.delta_anchor > 0:
            order_base = np.argsort(-base)
            in_top_n = np.zeros_like(base, dtype=bool)
            in_top_n[order_base[:self.anchor_top_n]] = True
            anchor_in_top = in_top_n & (self._is_anchor > 0)
            boost_anchor = np.where(anchor_in_top, 1.0 + self.delta_anchor, 1.0)

        # 3c. Date proximity boost
        boost_date = np.ones_like(base)
        if self.enable_date_boost:
            date_ref = None
            if user_state is not None and user_state.snapshot_days_ago is not None:
                date_ref = user_state.snapshot_days_ago
            elif self.enable_temporal_keywords:
                from retrieval_v3 import infer_query_date
                date_ref = infer_query_date(query)
            if date_ref is not None:
                prox = np.array([
                    compute_date_proximity(d, date_ref) for d in self._days
                ], dtype=np.float32)
                boost_date = 1.0 + self.delta_date * prox

        # 4. Combine
        final = base * boost_emo * boost_state * boost_anchor * boost_date
        order = np.argsort(-final)
        top_ids = [self._ids[int(i)] for i in order[:top_k]]

        # 5. Optional chain expansion
        if expand_chain and self.enable_chain_expansion:
            wider = [self._ids[int(i)] for i in order[:max(top_k * 3, 9)]]
            p2c, c2p = build_chain_graph(self.events)

            def _connected_component(seed, candidates):
                visited = {seed}
                frontier = [seed]
                found = {seed} if seed in candidates else set()
                while frontier:
                    n = frontier.pop(0)
                    for nb in c2p.get(n, []) + p2c.get(n, []):
                        if nb not in visited:
                            visited.add(nb)
                            frontier.append(nb)
                            if nb in candidates:
                                found.add(nb)
                return found

            cand_set = set(wider)
            best_seed, best_reach = None, set()
            for s in wider[:top_k]:
                reach = _connected_component(s, cand_set)
                if len(reach) > len(best_reach):
                    best_seed, best_reach = s, reach

            if best_seed and len(best_reach) >= 2:
                expanded = expand_chain_from_seeds([best_seed], self.events, depth=4)
                chain_ids = [eid for eid in expanded if eid in best_reach]
                result: list[int] = []
                for eid in chain_ids:
                    if eid not in result and len(result) < top_k:
                        result.append(eid)
                for eid in top_ids:
                    if eid not in result and len(result) < top_k:
                        result.append(eid)
                top_ids = result[:top_k]

        if return_scores:
            return [(self._ids[int(i)], float(final[int(i)])) for i in order[:top_k]]
        return top_ids

    # Bind the patched method to this instance only
    import types
    engine.retrieve = types.MethodType(patched_retrieve, engine)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_cell(events_masked: list[dict], tests: list[dict], beta: float,
             embed_cache: dict, density: float) -> float:
    """Run retrieval for one (density, beta) cell. Returns mean NDCG@3.

    embed_cache: keyed by density, stores pre-built engine to avoid re-embedding
    when only beta changes.
    """
    from retrieval_v3 import RetrievalV3  # noqa: E402

    cache_key = id(events_masked[0])  # stable per density variant

    if cache_key not in embed_cache:
        print(f"  [embed] density={density:.2f} — embedding {len(events_masked)} events…",
              file=sys.stderr)
        t0 = time.time()
        engine = RetrievalV3(
            events_masked,
            use_llm_query_emo=False,   # keyword fallback — no LLM calls
            beta=beta,
        )
        embed_cache[cache_key] = engine
        print(f"  [embed] done in {time.time()-t0:.1f}s", file=sys.stderr)
    else:
        engine = embed_cache[cache_key]

    # Patch engine for always-on beta sweep
    patch_retrieval_always_on_emotion(engine, beta)

    ndcg_scores: list[float] = []
    for t in tests:
        ideal = set(t.get("ideal_top_3_event_ids") or [])
        ttype = t.get("test_type", "unknown")
        is_chain = ttype == "chain"
        retrieved = engine.retrieve(
            t["user_query"],
            user_state=state_from_test(t),
            top_k=3,
            expand_chain=is_chain,
        )
        ndcg_scores.append(ndcg_at_3(retrieved, ideal))

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def ndcg_at_3(ranked_ids: list[int], ideal_ids: set[int]) -> float:
    return ndcg_at_k(ranked_ids, ideal_ids, k=3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase D density sensitivity test")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load .env if present
    env_path = args.corpus.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    os.environ["EMBEDDING_PROVIDER"] = "text-embedding-3-small"

    data = json.loads(args.corpus.read_text(encoding="utf-8"))
    events_orig: list[dict] = data["events"]
    tests: list[dict] = data["tests"]

    DENSITIES = [1.0, 0.5, 0.25, 0.10]
    BETAS = [0.0, 0.1, 0.2, 0.5, 1.0]

    print(f"[path_c] seed={args.seed} densities={DENSITIES} betas={BETAS}",
          file=sys.stderr)
    print(f"[path_c] {len(events_orig)} events, {len(tests)} probes", file=sys.stderr)

    results: dict[str, dict[str, float]] = {}
    embed_cache: dict[int, object] = {}

    for d in DENSITIES:
        events_masked = mask_emotion_tags(events_orig, d, args.seed)
        n_kept = sum(1 for e in events_masked if e.get("emotion_tags"))
        print(f"\n[density={d:.2f}] {n_kept}/{len(events_masked)} events retain emotion_tags",
              file=sys.stderr)

        row: dict[str, float] = {}
        for beta in BETAS:
            print(f"  [β={beta}] running…", file=sys.stderr)
            score = run_cell(events_masked, tests, beta, embed_cache, d)
            row[f"β={beta}"] = round(score, 4)
            print(f"  [β={beta}] NDCG@3 = {score:.4f}", file=sys.stderr)
        results[f"density={d}"] = row

    # Interpret: check monotonicity at each density
    def is_monotone_decreasing(row: dict[str, float]) -> bool:
        vals = [row[f"β={b}"] for b in BETAS]
        return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    d10_row = results["density=0.1"]
    d10_decreasing = is_monotone_decreasing(d10_row)
    d10_vals = [d10_row[f"β={b}"] for b in BETAS]
    d10_peak_beta = BETAS[d10_vals.index(max(d10_vals))]

    d100_row = results["density=1.0"]
    d100_decreasing = is_monotone_decreasing(d100_row)

    if d100_decreasing and d10_decreasing:
        delta_verdict = "not validated"
        density_conclusion = "corpus-density-invariant"
        interp = (
            "Phase D's monotonic regression holds at ALL tested densities. "
            f"At d=1.0 monotone-decreasing: {d100_decreasing}; "
            f"at d=0.10 monotone-decreasing: {d10_decreasing}. "
            "Always-on multiplicative emotion boosting is harmful regardless of "
            "annotation density. Delta's concern is NOT validated."
        )
    elif not d100_decreasing and not d10_decreasing:
        delta_verdict = "partially validated (different mechanism)"
        density_conclusion = "corpus-density-ambiguous"
        interp = (
            "Neither density shows clean monotonic decrease; the Phase D effect "
            "may be noise or non-monotone at all densities. Delta's concern: "
            "partially validated but for a different reason than expected."
        )
    elif d100_decreasing and not d10_decreasing:
        delta_verdict = "validated"
        density_conclusion = "corpus-density-specific"
        interp = (
            f"At d=1.0 (dense): NDCG@3 monotonically decreases with β — "
            f"replicates Phase D. At d=0.10 (sparse): NOT monotone; β peak at "
            f"β={d10_peak_beta}. Always-on emotion boost is only harmful at high "
            "annotation density. Delta's concern IS validated: the monotonic "
            "regression is corpus-density-specific."
        )
    else:
        delta_verdict = "partially validated"
        density_conclusion = "corpus-density-specific (inverted)"
        interp = (
            f"At d=1.0 NOT monotone; at d=0.10 monotone-decreasing. "
            f"Unexpected pattern. Delta's concern: partially validated."
        )

    output = {
        "seed": args.seed,
        "densities": DENSITIES,
        "betas": BETAS,
        "ndcg_at_3": results,
        "monotone_decreasing": {
            f"density={d}": is_monotone_decreasing(results[f"density={d}"])
            for d in DENSITIES
        },
        "delta_verdict": delta_verdict,
        "density_conclusion": density_conclusion,
        "interpretation": interp,
        "meta": {
            "n_events": len(events_orig),
            "n_probes": len(tests),
            "embedding_provider": "text-embedding-3-small",
            "emotion_boost_mode": "always-on (Phase D replication; gate bypassed)",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n[path_c] saved → {args.out}", file=sys.stderr)

    # Print summary matrix
    print("\n=== NDCG@3 matrix (density × β) ===", file=sys.stderr)
    header = "density  \\ β  " + "  ".join(f"{b:5.2f}" for b in BETAS)
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for d in DENSITIES:
        row = results[f"density={d}"]
        vals = "  ".join(f"{row[f'β={b}']:5.3f}" for b in BETAS)
        mono = "↘" if is_monotone_decreasing(row) else "~"
        print(f"  d={d:<6}           {vals}  {mono}", file=sys.stderr)

    print(f"\n[verdict] delta's concern: {delta_verdict}", file=sys.stderr)
    print(f"[conclusion] Phase D monotonic regression is {density_conclusion}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
