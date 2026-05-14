"""§5.X ablation: Pulse v3 stateful probes (T6-T15) with real Apple Health biometric
snapshots (5 best + 5 depleted from year+ deployment) replacing the synthetic
biometric component of author-authored state. mood_vector preserved (load-bearing
for A/B pair design); only sleep_quality/sleep_hours/hrv/hr_trend/stress_proxy
swapped with real values.

Design:
- A probes (T6, T8, T10, T12, T14) get one of 5 resourceful real snapshots
- B probes (T7, T9, T11, T13, T15) get one of 5 depleted real snapshots
- Stateful R@3 measured per probe + aggregate
- Comparison: baseline (synthetic biometric, current paper) vs real-biometric

Output: external-evals/results/pulse_v3-real-biometric-ablation-{ts}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


def recall_at_3(retrieved, ideal):
    if not ideal: return 0.0
    return len(set(retrieved[:3]) & set(ideal)) / len(ideal)


def overlay_real_biometric(probe_state, real_snap):
    """Take probe's mood_vector but replace biometric with real snap."""
    return {
        "mood_vector": probe_state.get("mood_vector", {}),
        "sleep_quality": real_snap["sleep_quality"],
        "sleep_hours": real_snap["sleep_hours"],
        "hrv": real_snap["hrv"],
        "hr_trend": real_snap["hr_trend"],
        "stress_proxy": real_snap["stress_proxy"],
        "time_of_day": real_snap["time_of_day"],
        "recent_life_events_7d": probe_state.get("recent_life_events_7d", []),
        "snapshot_days_ago": real_snap["snapshot_days_ago"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--real-snapshots", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embedding", default="cohere", choices=["cohere", "text-embedding-3-small"])
    args = ap.parse_args()

    os.environ["EMBEDDING_PROVIDER"] = args.embedding
    from retrieval_v3 import RetrievalV3, UserState

    corpus = json.loads(args.corpus.read_text())
    snaps = json.loads(args.real_snapshots.read_text())

    events = corpus["events"]
    stateful = [t for t in corpus["tests"] if t["test_type"] == "stateful"]
    # Sort by id to ensure A/B ordering
    stateful.sort(key=lambda t: int(t["id"][1:]) if t["id"].startswith("T") else 999)
    print(f"[real-biometric] {len(stateful)} stateful probes, {args.embedding} backbone", file=sys.stderr)
    print(f"[real-biometric] {len(snaps['resourceful'])} resourceful + {len(snaps['depleted'])} depleted snapshots", file=sys.stderr)

    t0 = time.time()
    engine = RetrievalV3(events, use_llm_query_emo=False)
    print(f"[real-biometric] events embedded in {time.time()-t0:.1f}s", file=sys.stderr)

    res_snaps = snaps["resourceful"]  # for A probes (good state queries)
    dep_snaps = snaps["depleted"]      # for B probes (depleted state queries)

    def make_state(s):
        if not s: return None
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
    baseline_r3 = []
    real_r3 = []
    for ti, t in enumerate(stateful):
        ideal = set(t.get("ideal_top_3_event_ids") or [])
        # A or B by name suffix
        is_A = t["name"].endswith("_A")
        snap_pool = res_snaps if is_A else dep_snaps
        # Round-robin within pool
        snap_idx = (ti // 2) % len(snap_pool)
        snap = snap_pool[snap_idx]

        baseline_state = t.get("user_state") or {}
        real_state = overlay_real_biometric(baseline_state, snap)

        # Baseline retrieve (synthetic biometric)
        ids_baseline = engine.retrieve(
            t["user_query"],
            user_state=make_state(baseline_state),
            top_k=5,
            expand_chain=False,
            return_scores=True,
        )
        retrieved_baseline = [int(eid) for eid, _ in ids_baseline][:5]
        r3_b = recall_at_3(retrieved_baseline, ideal)

        # Real biometric retrieve
        ids_real = engine.retrieve(
            t["user_query"],
            user_state=make_state(real_state),
            top_k=5,
            expand_chain=False,
            return_scores=True,
        )
        retrieved_real = [int(eid) for eid, _ in ids_real][:5]
        r3_r = recall_at_3(retrieved_real, ideal)

        baseline_r3.append(r3_b)
        real_r3.append(r3_r)

        per_test.append({
            "test_id": t["id"],
            "name": t["name"],
            "ideal_top_3": sorted(ideal),
            "synthetic_state_mood": baseline_state.get("mood_vector"),
            "synthetic_state_biometric": {k: baseline_state.get(k) for k in ["sleep_quality", "sleep_hours", "hrv", "hr_trend", "stress_proxy"]},
            "real_state_biometric": {k: real_state.get(k) for k in ["sleep_quality", "sleep_hours", "hrv", "hr_trend", "stress_proxy"]},
            "real_snapshot_date": snap["date"],
            "retrieved_baseline": retrieved_baseline,
            "retrieved_real": retrieved_real,
            "r3_baseline": r3_b,
            "r3_real": r3_r,
            "delta_r3": r3_r - r3_b,
        })
        print(f"  [{ti+1}/{len(stateful)}] {t['name']:40s}  baseline R@3={r3_b:.2f}  real={r3_r:.2f}  Δ={r3_r-r3_b:+.2f}", file=sys.stderr)

    mean_baseline = sum(baseline_r3) / len(baseline_r3)
    mean_real = sum(real_r3) / len(real_r3)

    summary = {
        "n_stateful_probes": len(stateful),
        "embedding": args.embedding,
        "snapshots_used": {
            "resourceful_dates": [s["date"] for s in res_snaps],
            "depleted_dates": [s["date"] for s in dep_snaps],
        },
        "baseline_synthetic_stateful_r3": mean_baseline,
        "real_biometric_stateful_r3": mean_real,
        "delta_r3": mean_real - mean_baseline,
        "interpretation": (
            "If delta is near zero, Pulse stateful R@3 is robust to swapping synthetic biometric with real Apple Health values "
            "(closes 'state synthetic' peer-review concern). If delta is positive, real biometric helps; if strongly negative, "
            "synthetic biometric was helping artificially and the architecture needs honest disclosure."
        ),
        "per_test": per_test,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[summary] baseline (synthetic biometric) mean stateful R@3 = {mean_baseline:.4f}", file=sys.stderr)
    print(f"[summary] real biometric            mean stateful R@3 = {mean_real:.4f}", file=sys.stderr)
    print(f"[summary] delta                                          = {mean_real - mean_baseline:+.4f}", file=sys.stderr)
    print(f"[save] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
