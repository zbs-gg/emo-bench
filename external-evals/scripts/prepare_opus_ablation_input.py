"""Generate Opus-in-chat voting input for ablation runs.

For each of the 7 ablation snapshots (v2_pure, no_boosts, anchor_only,
emotion_only, date_only, state_only, full) reads result.json, extracts
the retrieved event-ids per system per test, joins with the event text
from corpus-v3.json, and writes a single consolidated JSON file that
Opus-in-chat can read batch-by-batch.

Usage:
  python prepare_opus_ablation_input.py \\
      --snapshots-dir external-evals/snapshots \\
      --prefix 2026-04-25-ablation- \\
      --out /tmp/opus_ablation_input.json

Output schema:
  {
    "_meta": {...},
    "configs": ["v2_pure", "no_boosts", ...],
    "batches": [
      {
        "config": "v2_pure",
        "test_id": "T1",
        "test_name": "cold_open_salience",
        "test_type": "core",
        "user_query": "...",
        "user_state": {...} | null,
        "biometric_snapshot": {...} | null,
        "ideal_top_3_event_ids": [5, 2, 20],
        "retrievals": {
          "cosine": [{"id": 5, "text": "..."}, ...],
          "bm25":   [...],
          "hybrid": [...],
          "pulse_v3": [...]
        }
      },
      ...
    ]
  }

One entry per (config, test) pair. 7 configs × 35 tests = 245 entries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIGS = ["v2_pure", "no_boosts", "anchor_only", "emotion_only",
           "date_only", "state_only", "full"]


def load_corpus_events(corpus_path: Path) -> dict[int, dict]:
    """Return {event_id → event_dict} for quick text lookup."""
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    return {e["id"]: e for e in data["events"]}


def load_corpus_tests(corpus_path: Path) -> dict[str, dict]:
    """Return {test_id → test_dict}."""
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    return {t["id"]: t for t in data["tests"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots-dir", type=Path,
                    default=Path("external-evals/snapshots"))
    ap.add_argument("--prefix", type=str, default="2026-04-25-ablation-")
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/opus_ablation_input.json"))
    ap.add_argument("--corpus", type=Path,
                    default=Path("datasets/empathic-memory-corpus-v3.json"))
    args = ap.parse_args()

    events_by_id = load_corpus_events(args.corpus)
    tests_by_id = load_corpus_tests(args.corpus)

    batches = []
    configs_found = []
    for cfg in CONFIGS:
        snap_dir = args.snapshots_dir / f"{args.prefix}{cfg}"
        rp = snap_dir / "result.json"
        if not rp.exists():
            print(f"[skip] missing: {rp}")
            continue
        configs_found.append(cfg)
        data = json.loads(rp.read_text(encoding="utf-8"))
        for row in data.get("tests", []):
            tid = row["test_id"]
            test = tests_by_id.get(tid, {})
            batches.append({
                "config": cfg,
                "test_id": tid,
                "test_name": row.get("test_name"),
                "test_type": row.get("test_type"),
                "user_query": row.get("user_query"),
                "user_state": test.get("user_state") or test.get("user_state_overlay"),
                "biometric_snapshot": test.get("biometric_snapshot"),
                "ideal_top_3_event_ids": test.get("ideal_top_3_event_ids"),
                "ideal_chain": test.get("ideal_chain"),
                "retrievals": {
                    sn: [
                        {
                            "id": eid,
                            "text": (events_by_id.get(eid, {}).get("text") or "")[:400],
                            "days_ago": events_by_id.get(eid, {}).get("days_ago"),
                            "sentiment_label": events_by_id.get(eid, {})
                                                            .get("sentiment_label"),
                            "user_flag": events_by_id.get(eid, {}).get("user_flag"),
                        }
                        for eid in row.get("retrievals", {}).get(sn, [])
                    ]
                    for sn in row.get("retrievals", {})
                },
            })

    out = {
        "_meta": {
            "purpose": "Opus-in-chat voting input for 7-config ablation",
            "configs": configs_found,
            "n_batches": len(batches),
        },
        "batches": batches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[write] {args.out}")
    print(f"  configs: {len(configs_found)} ({', '.join(configs_found)})")
    print(f"  batches: {len(batches)}")
    print(f"  size:    {args.out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
