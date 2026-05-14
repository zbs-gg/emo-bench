"""Aggregate 7 ablation snapshots into a single matrix + paper table.

Reads snapshots produced by
    for CFG in v2_pure no_boosts anchor_only emotion_only date_only state_only full; do
        bench-empathic-memory-v3.py --ablation-config "$CFG" \\
            --snapshot "2026-04-25-ablation-${CFG}" ...
    done

Writes:
  - snapshots/2026-04-25-ablation-matrix/ablation-matrix.json  (webapp feed)
  - snapshots/2026-04-25-ablation-matrix/ablation.md           (paper table)
  - webapp-next/public/data/ablation-matrix.json               (fast-path for Next.js)

Usage:
  python aggregate_ablation.py \\
      --snapshots-dir external-evals/snapshots \\
      --prefix 2026-04-25-ablation- \\
      --out-snapshot external-evals/snapshots/2026-04-25-ablation-matrix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFIGS = ["v2_pure", "no_boosts", "anchor_only", "emotion_only",
           "date_only", "state_only", "full"]

AXES = ["overall_weighted_mean", "core_mean", "stateful_mean",
        "chain_mean", "multi_signal_mean"]

AXIS_LABEL = {
    "overall_weighted_mean": "overall",
    "core_mean": "core",
    "stateful_mean": "stateful",
    "chain_mean": "chain",
    "multi_signal_mean": "multi",
}


def load_one(snap_dir: Path) -> dict | None:
    result_path = snap_dir / "result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text())


def build_matrix(snapshots_dir: Path, prefix: str) -> dict:
    matrix: dict = {
        "configs": [],
        "systems_seen": set(),
        "per_config": {},  # config -> {system -> {axis -> value}}
    }
    for cfg in CONFIGS:
        snap = snapshots_dir / f"{prefix}{cfg}"
        data = load_one(snap)
        if data is None:
            print(f"[skip] missing snapshot: {snap}")
            continue
        matrix["configs"].append(cfg)
        per_sys: dict[str, dict[str, float | None]] = {}
        for sn, agg in data.get("aggregate", {}).items():
            matrix["systems_seen"].add(sn)
            per_sys[sn] = {AXIS_LABEL[a]: agg.get(a) for a in AXES}
            per_sys[sn]["n_tests"] = agg.get("n_tests")
        matrix["per_config"][cfg] = per_sys

    matrix["systems_seen"] = sorted(matrix["systems_seen"])
    return matrix


def contribution_table(matrix: dict, target_system: str = "pulse_v3") -> list[dict]:
    """Per-axis contribution: (isolated - v2_pure) for each isolated config.
    Only meaningful for target_system (pulse_v3), since baselines don't change.
    """
    v2 = matrix["per_config"].get("v2_pure", {}).get(target_system, {})
    rows = []
    for cfg in matrix["configs"]:
        cur = matrix["per_config"].get(cfg, {}).get(target_system, {})
        row = {"config": cfg}
        for axis in ("overall", "core", "stateful", "chain", "multi"):
            cv = cur.get(axis)
            bv = v2.get(axis)
            if cv is None or bv is None:
                row[axis] = None
                row[f"Δ{axis}"] = None
            else:
                row[axis] = round(float(cv), 3)
                row[f"Δ{axis}"] = round(float(cv) - float(bv), 3)
        rows.append(row)
    return rows


def render_markdown(matrix: dict) -> str:
    rows = contribution_table(matrix, "pulse_v3")
    lines = [
        "# Ablation study — empathic-memory-bench v3",
        "",
        "Pulse v3 under each ablation preset. Δ-column is the change in the "
        "axis vs `v2_pure` (all boosts OFF).",
        "",
        "## Overall scores (pulse_v3)",
        "",
        "| config | overall | Δoverall | core | Δcore | stateful | Δstateful | chain | Δchain | multi | Δmulti |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        def _f(x):
            return f"{x:+.2f}" if isinstance(x, (int, float)) else "—"

        def _v(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "—"

        lines.append(
            f"| {r['config']} | {_v(r['overall'])} | {_f(r['Δoverall'])} | "
            f"{_v(r['core'])} | {_f(r['Δcore'])} | "
            f"{_v(r['stateful'])} | {_f(r['Δstateful'])} | "
            f"{_v(r['chain'])} | {_f(r['Δchain'])} | "
            f"{_v(r['multi'])} | {_f(r['Δmulti'])} |"
        )

    lines.extend([
        "",
        "## Sanity",
        "",
        "- `v2_pure` and `no_boosts` should produce ≈identical scores (integrity alias).",
        "- Every positive Δ on the stateful/multi axis is the isolated boost's direct contribution.",
        "- Sum of isolated Δs ≠ full Δ in general — boosts interact multiplicatively.",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots-dir", type=Path,
                    default=Path("external-evals/snapshots"))
    ap.add_argument("--prefix", type=str, default="2026-04-25-ablation-")
    ap.add_argument("--out-snapshot", type=Path,
                    default=Path("external-evals/snapshots/2026-04-25-ablation-matrix"))
    ap.add_argument("--webapp-data", type=Path,
                    default=Path("webapp-next/public/data/ablation-matrix.json"),
                    help="Secondary output for Next.js static data")
    args = ap.parse_args()

    matrix = build_matrix(args.snapshots_dir, args.prefix)
    if not matrix["configs"]:
        raise SystemExit(f"No ablation snapshots found under {args.snapshots_dir} "
                         f"with prefix {args.prefix}")

    # Add contribution rows for convenience
    matrix["contributions_pulse_v3"] = contribution_table(matrix, "pulse_v3")
    matrix["target_system"] = "pulse_v3"
    from datetime import datetime, timezone
    matrix["generated_at"] = datetime.now(timezone.utc).isoformat()

    args.out_snapshot.mkdir(parents=True, exist_ok=True)
    (args.out_snapshot / "ablation-matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2)
    )
    (args.out_snapshot / "ablation.md").write_text(render_markdown(matrix))
    print(f"[snapshot] {args.out_snapshot}/")

    if args.webapp_data:
        args.webapp_data.parent.mkdir(parents=True, exist_ok=True)
        args.webapp_data.write_text(json.dumps(matrix, ensure_ascii=False, indent=2))
        print(f"[webapp]   {args.webapp_data}")


if __name__ == "__main__":
    main()
