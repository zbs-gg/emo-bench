"""Deduplicate Opus voting workload across ablation configs.

Many configs share retrievals on tests where their boosts don't activate
(e.g. core tests without user_state see identical retrievals from all 7 configs).
This script:

1. Reads /tmp/opus_ablation_input.json (245 batches across 7 configs).
2. Groups batches by (test_id, retrieval_signature). Signature is the tuple
   of retrieved event-id lists across all 4 systems for that batch.
3. Pre-fills verdicts for "full" config from existing main-run Opus verdicts
   (opus_verdicts_2026-04-24_v2.json) when retrievals match.
4. Outputs:
   - /tmp/opus_unique_batches.json — unique batches still needing a vote
   - /tmp/opus_replication_map.json — maps unique-batch-id → list of
     (config, test_id) tuples to replicate the verdict to.

Usage:
  python dedupe_opus_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

INPUT = Path("/tmp/opus_ablation_input.json")
EXISTING_OPUS = Path(
    "/Users/nikshilov/dev/ai/Garden/bench/external-evals/results/"
    "opus_verdicts_2026-04-24_v2.json"
)
OUT_UNIQUE = Path("/tmp/opus_unique_batches.json")
OUT_REPLICATION = Path("/tmp/opus_replication_map.json")
OUT_PREFILLED = Path("/tmp/opus_ablation_verdicts.json")


def signature(batch: dict) -> str:
    """Stable string signature of the 4-system retrieval shapes for a batch."""
    parts = []
    for sn in sorted(batch.get("retrievals", {}).keys()):
        ids = [item.get("id") for item in batch["retrievals"][sn]]
        parts.append(f"{sn}:{','.join(str(i) for i in ids)}")
    return "|".join(parts)


def main():
    data = json.loads(INPUT.read_text())
    batches = data["batches"]
    print(f"Loaded {len(batches)} batches from {INPUT}")

    # Step 1: group by (test_id, signature)
    groups: dict[tuple[str, str], list[dict]] = {}
    for b in batches:
        key = (b["test_id"], signature(b))
        groups.setdefault(key, []).append(b)
    print(f"Unique (test_id, retrieval_signature) groups: {len(groups)}")

    # Step 2: load existing Opus verdicts from main run
    existing = {}
    if EXISTING_OPUS.exists():
        v = json.loads(EXISTING_OPUS.read_text())
        existing = v.get("verdicts", {})
        print(f"Existing main-run Opus verdicts: {len(existing)} test_ids "
              f"(from {EXISTING_OPUS.name})")

    # Step 3: build prefilled verdicts for any group whose retrieval signature
    # matches the main-run signature. Main-run was full config — use it as
    # ground truth for any group containing 'full' OR any group that happens
    # to share signature with full's retrievals on the same test.
    prefilled: dict[str, dict[str, dict]] = {cfg: {} for cfg in [
        "v2_pure", "no_boosts", "anchor_only", "emotion_only",
        "date_only", "state_only", "full"
    ]}
    matched_via_full = 0

    # Find each test's "full" signature (representative of main run's retrievals)
    full_sig_by_test: dict[str, str] = {}
    for b in batches:
        if b["config"] == "full":
            full_sig_by_test[b["test_id"]] = signature(b)

    # For every group, if its signature == full's signature for that test_id,
    # we can reuse the existing main-run Opus verdict.
    unique_batches = []  # batches still needing a vote
    replication: dict[str, list[tuple[str, str]]] = {}

    for (tid, sig), grp in groups.items():
        # Replication targets — every (config, test_id) covered by this group
        targets = [(b["config"], b["test_id"]) for b in grp]

        full_sig = full_sig_by_test.get(tid)
        if full_sig == sig and tid in existing:
            # Pre-fill from main run
            for cfg, t in targets:
                prefilled[cfg][t] = existing[tid]
            matched_via_full += len(targets)
            continue

        # Otherwise this group needs an Opus vote
        # Pick one representative batch
        rep = grp[0]
        unique_id = f"{tid}__{sig[:30].replace(',', '-').replace(':', '_')}"
        rep_copy = dict(rep)
        rep_copy["__unique_id"] = unique_id
        rep_copy["__replicate_to"] = targets
        unique_batches.append(rep_copy)
        replication[unique_id] = targets

    # Save outputs
    OUT_UNIQUE.parent.mkdir(parents=True, exist_ok=True)
    OUT_UNIQUE.write_text(json.dumps({
        "_meta": {
            "n_unique_batches": len(unique_batches),
            "n_total_batches": len(batches),
            "n_prefilled_via_full": matched_via_full,
        },
        "batches": unique_batches,
    }, ensure_ascii=False, indent=2))
    OUT_REPLICATION.write_text(json.dumps(replication, ensure_ascii=False, indent=2))
    OUT_PREFILLED.write_text(json.dumps({
        "_meta": {
            "judge": "opus-4.7-in-chat",
            "source": "claude-opus-4-7[1m] via Claude Code chat",
            "prefilled_from": str(EXISTING_OPUS.name),
        },
        "verdicts": prefilled,
    }, ensure_ascii=False, indent=2))

    print(f"\n[unique]      {OUT_UNIQUE}  ({len(unique_batches)} batches needing a vote)")
    print(f"[replication] {OUT_REPLICATION}")
    print(f"[prefilled]   {OUT_PREFILLED}  ({matched_via_full} verdicts reused from main run)")
    print(f"\nWorkload reduction: {len(batches)} → {len(unique_batches)} "
          f"({100*(1 - len(unique_batches)/len(batches)):.0f}% saved)")


if __name__ == "__main__":
    main()
