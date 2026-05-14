"""Merge Opus-in-chat verdicts into the 7 ablation snapshot result.json files.

Input: /tmp/opus_ablation_verdicts.json produced by manually voting through
the batches in /tmp/opus_ablation_input.json. Format:

  {
    "_meta": {"judge": "opus-4.7-in-chat", ...},
    "verdicts": {
      "v2_pure": {
        "T1": {
          "cosine_rel": 7, "cosine_spec": 7, "cosine_act": 7,
          "bm25_rel": 4, ..., "winner": "cosine", "note": "..."
        },
        "T2": {...}, ...
      },
      "no_boosts": {...},
      ...
    }
  }

For each config, this script injects Opus as the 8th judge into that config's
result.json tests[].verdicts map and re-runs the per-test aggregation so the
aggregate row picks up the 8-judge mean.

Usage:
  python merge_opus_into_ablation.py \\
      --verdicts /tmp/opus_ablation_verdicts.json \\
      --snapshots-dir external-evals/snapshots \\
      --prefix 2026-04-25-ablation-

Writes each snapshot's result.json in place (backup kept as result.json.bak).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


# Axis weights for per-test aggregate (mirrors bench-empathic-memory-v3.py)
def _weighted_test_score(test_type, core_sum, stateful, chain, multi):
    """Same formula as bench-empathic-memory-v3.py::weighted_test_score."""
    if test_type == "core":
        return (core_sum / 3.0) if core_sum is not None else 0.0
    if test_type == "stateful":
        c = (core_sum / 3.0) if core_sum is not None else 0.0
        s = stateful if stateful is not None else 0.0
        return 0.70 * c + 0.30 * s
    if test_type == "chain":
        c = (core_sum / 3.0) if core_sum is not None else 0.0
        ch = chain if chain is not None else 0.0
        return 0.50 * ch + 0.50 * c
    if test_type == "multi_signal":
        c = (core_sum / 3.0) if core_sum is not None else 0.0
        m = multi if multi is not None else 0.0
        return 0.60 * m + 0.40 * c
    return 0.0


def merge_one(snap_path: Path, opus_verdicts_for_config: dict) -> tuple[int, int]:
    """Inject opus verdicts + recompute aggregates. Returns (n_added, n_tests)."""
    result_path = snap_path / "result.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))

    # Backup
    bak = snap_path / "result.json.bak"
    if not bak.exists():
        shutil.copy(result_path, bak)

    # Add opus to judge list in _meta
    judges = data.setdefault("_meta", {}).setdefault("judges", [])
    if "opus" not in judges:
        judges.append("opus")

    per_system_scores: dict[str, list[float]] = {}
    per_system_axes: dict[str, dict[str, list[float]]] = {}
    added = 0

    for row in data.get("tests", []):
        tid = row["test_id"]
        tt = row.get("test_type", "core")
        systems = list(row.get("scores", {}).keys())
        # Fallback if `scores` empty: infer systems from retrievals
        if not systems:
            systems = list(row.get("retrievals", {}).keys())

        # Inject opus verdict if present
        opus_v = opus_verdicts_for_config.get(tid)
        if opus_v is not None:
            row.setdefault("verdicts", {})["opus"] = opus_v
            added += 1

        # Recompute per-system scores using ALL judges (including opus now)
        vlist = row.get("verdicts", {})
        per_sys = {}
        for sn in systems:
            rels, specs, acts, statefuls, multis = [], [], [], [], []
            for j, v in vlist.items():
                if "error" in v:
                    continue
                if f"{sn}_rel" in v:
                    rels.append(float(v[f"{sn}_rel"]))
                if f"{sn}_spec" in v:
                    specs.append(float(v[f"{sn}_spec"]))
                if f"{sn}_act" in v:
                    acts.append(float(v[f"{sn}_act"]))
                if f"{sn}_stateful" in v:
                    statefuls.append(float(v[f"{sn}_stateful"]))
                if f"{sn}_multi_signal" in v:
                    multis.append(float(v[f"{sn}_multi_signal"]))
            core_sum = None
            if rels and specs and acts:
                core_sum = float(np.mean(rels) + np.mean(specs) + np.mean(acts))
            stateful_v = float(np.mean(statefuls)) if statefuls else None
            multi_v = float(np.mean(multis)) if multis else None
            chain_v = row.get("scores", {}).get(sn, {}).get("chain")
            score = _weighted_test_score(tt, core_sum, stateful_v, chain_v, multi_v)
            per_sys[sn] = {
                "rel": float(np.mean(rels)) if rels else None,
                "spec": float(np.mean(specs)) if specs else None,
                "act": float(np.mean(acts)) if acts else None,
                "core_sum": core_sum,
                "stateful": stateful_v,
                "chain": chain_v,
                "multi_signal": multi_v,
                "weighted": score,
            }
            per_system_scores.setdefault(sn, []).append(score)
            if tt in ("core", "multi_signal", "stateful") and core_sum is not None:
                per_system_axes.setdefault(sn, {}).setdefault("core", []).append(core_sum / 3.0)
            if tt == "stateful" and stateful_v is not None:
                per_system_axes.setdefault(sn, {}).setdefault("stateful", []).append(stateful_v)
            if tt == "chain" and chain_v is not None:
                per_system_axes.setdefault(sn, {}).setdefault("chain", []).append(chain_v)
            if tt == "multi_signal" and multi_v is not None:
                per_system_axes.setdefault(sn, {}).setdefault("multi_signal", []).append(multi_v)
        row["scores"] = per_sys

    # Recompute aggregate
    out_aggregate = {}
    for sn in per_system_scores:
        scores = per_system_scores[sn]
        ax = per_system_axes.get(sn, {})
        out_aggregate[sn] = {
            "overall_weighted_mean": float(np.mean(scores)) if scores else 0.0,
            "n_tests": len(scores),
            "core_mean": float(np.mean(ax.get("core", []))) if ax.get("core") else None,
            "stateful_mean": float(np.mean(ax.get("stateful", []))) if ax.get("stateful") else None,
            "chain_mean": float(np.mean(ax.get("chain", []))) if ax.get("chain") else None,
            "multi_signal_mean": float(np.mean(ax.get("multi_signal", []))) if ax.get("multi_signal") else None,
        }
    data["aggregate"] = out_aggregate

    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return added, len(data.get("tests", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", type=Path, required=True,
                    help="Opus verdicts JSON (see module docstring for format)")
    ap.add_argument("--snapshots-dir", type=Path,
                    default=Path("external-evals/snapshots"))
    ap.add_argument("--prefix", type=str, default="2026-04-25-ablation-")
    args = ap.parse_args()

    v = json.loads(args.verdicts.read_text(encoding="utf-8"))
    verdicts_by_config = v.get("verdicts", {})

    total_added = 0
    for cfg, per_test_verdicts in verdicts_by_config.items():
        snap = args.snapshots_dir / f"{args.prefix}{cfg}"
        if not snap.exists():
            print(f"[skip] no snapshot for config={cfg}: {snap}")
            continue
        added, n_tests = merge_one(snap, per_test_verdicts)
        print(f"[merged] {cfg}: +{added}/{n_tests} opus verdicts → {snap / 'result.json'}")
        total_added += added

    print(f"\nTotal opus verdicts merged: {total_added}")


if __name__ == "__main__":
    main()
