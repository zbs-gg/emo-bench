"""Replicate Opus verdicts from /tmp/opus_unique_verdicts.json across all
configs in the ablation matrix using /tmp/opus_replication_map.json.

Pre-fills any tests already in /tmp/opus_ablation_verdicts.json (from main
8-judge run reuse) and adds the newly-voted unique batches.

Output: /tmp/opus_ablation_verdicts.json with all 7 configs × 35 tests filled.
"""
from __future__ import annotations

import json
from pathlib import Path

UNIQUE = Path("/tmp/opus_unique_verdicts.json")
REPLICATION = Path("/tmp/opus_replication_map.json")
PREFILLED = Path("/tmp/opus_ablation_verdicts.json")


def main():
    # Load existing pre-filled verdicts (from main 8-judge run reuse)
    pre = json.loads(PREFILLED.read_text())
    verdicts_by_config: dict[str, dict[str, dict]] = pre.get("verdicts", {})

    # Load fresh-voted unique verdicts
    unique = json.loads(UNIQUE.read_text())
    unique_verdicts = unique["verdicts"]

    # Load replication map: unique_id -> [(config, test_id), ...]
    replication = json.loads(REPLICATION.read_text())

    # Replicate each unique verdict to all configs in its target list
    n_replicated = 0
    n_unique_voted = 0
    for unique_id, targets in replication.items():
        verdict = unique_verdicts.get(unique_id)
        if verdict is None:
            print(f"[WARN] no verdict for unique_id {unique_id} (skipping {len(targets)} targets)")
            continue
        n_unique_voted += 1
        for cfg, tid in targets:
            verdicts_by_config.setdefault(cfg, {})[tid] = verdict
            n_replicated += 1

    # Save merged verdicts
    out = {
        "_meta": {
            "judge": "opus-4.7-in-chat",
            "source": "claude-opus-4-7[1m] via Claude Code chat",
            "n_unique_voted": n_unique_voted,
            "n_total_replicated": n_replicated,
            "n_prefilled_from_main_run": pre.get("_meta", {}).get("prefilled_from", "unknown"),
        },
        "verdicts": verdicts_by_config,
    }
    PREFILLED.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    # Verify coverage
    print(f"\n[summary]")
    print(f"  unique verdicts voted: {n_unique_voted}")
    print(f"  total replicated to (config, test): {n_replicated}")
    print()
    for cfg in sorted(verdicts_by_config.keys()):
        n = len(verdicts_by_config[cfg])
        print(f"  {cfg:14s}: {n:2d}/35 verdicts")


if __name__ == "__main__":
    main()
