#!/usr/bin/env python3
"""Rebuild summary.json + report.md from raw/<judge>.json after manual patches."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"

if len(sys.argv) < 2:
    sys.exit("usage: bench-pulse-rebuild-report.py <run-dir-name>")

RUN_DIR = RESULTS_DIR / sys.argv[1]
RAW = RUN_DIR / "raw"

JUDGE_ORDER = [
    ("opus",     "Opus 4.6"),
    ("sonnet",   "Sonnet 4.6"),
    ("haiku",    "Haiku 4.5"),
    ("glm5",     "GLM-5.1"),
    ("deepseek", "DeepSeek V3.2"),
    ("qwen",     "Qwen 3.6 Plus"),
    ("kimi",     "Kimi K2.5"),
    ("grok",     "Grok 4.20"),
    ("gpt4o",    "GPT-4o"),
    ("gpt54",    "GPT-5.4"),
    ("gpt54pro", "GPT-5.4 Pro"),
    ("gemini",   "Gemini 3.1 Pro"),
]

SCORE_KEYS = [
    "1_architecture",
    "2_triage_prompt",
    "3_extract_prompt",
    "4_resolver",
    "5_scorer",
    "6_apply_to_graph",
    "7_cost_efficiency",
]


def _pick_int(x):
    try:
        if isinstance(x, bool):
            return None
        return int(x)
    except Exception:
        return None


def _collect_score(verdicts, key):
    vals = []
    for v in verdicts:
        scores = v.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        n = _pick_int(scores.get(key))
        if n is None:
            # try without numeric prefix ("architecture", etc.)
            tail = key.split("_", 1)[1] if "_" in key else key
            n = _pick_int(scores.get(tail))
        if n is not None and 1 <= n <= 10:
            vals.append(n)
    avg = sum(vals) / len(vals) if vals else None
    return vals, avg


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "with", "for",
    "from", "by", "is", "are", "be", "no", "not", "do", "does", "did", "can",
    "this", "that", "these", "those", "it", "its", "as", "at", "so", "up", "off",
    "if", "then", "than", "into", "onto", "via", "per", "all", "any", "some",
    "other", "one", "two", "three", "four", "will", "must", "should", "could",
    "there", "their", "they", "them", "our", "we", "you", "your", "i", "me",
    "every", "each", "also", "even", "still", "currently", "only", "just", "yet",
}


def _tokens(s):
    toks = re.findall(r"[a-zA-Zа-яА-Я][a-zA-Zа-яА-Я_]{2,}", s.lower())
    return {t for t in toks if t not in _STOPWORDS}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# Theme keywords: if an item matches >=2 keywords of a theme, bucket it there
_THEMES = [
    ("No per-observation isolation — one crash rolls back the whole batch",
     ["observation", "isolation", "batch", "rollback", "rolls", "savepoint",
      "transaction", "crash", "poison", "kills", "failure", "cascade", "atomic",
      "wrapped", "single", "malformed"]),
    ("Missing UNIQUE constraints — relations/facts/events accumulate duplicates",
     ["unique", "constraint", "duplicate", "dupe", "duplication", "bloat",
      "dedup", "accumulate", "unbounded", "rerun", "retry", "upsert", "idempot"]),
    ("Events dropped from graph — entities-involved field ignored by apply stage",
     ["events", "event", "orphan", "disconnected", "involved", "linkage", "junction",
      "link", "participants", "entity_ids", "dropped"]),
    ("Cross-script / transliteration: Cyrillic vs Latin entity duplication",
     ["transliteration", "cyrillic", "latin", "russian", "cross-script", "nickname",
      "аня", "anya", "anna", "ник", "nikita", "elle", "элли", "unicode", "nfkc",
      "diacritic", "script"]),
    ("No JSON schema validation before apply — canonical_name and kind fields not enforced",
     ["schema", "validation", "validate", "jsonschema", "canonical_name", "validator",
      "enforce", "enforcement", "contract", "required field", "json-schema"]),
    ("Use tool-use / structured output instead of free-form JSON",
     ["tool-use", "tool use", "function-calling", "function calling", "structured output",
      "response_format", "json mode", "json_mode", "tool_use", "strict mode"]),
    ("Add concrete examples to triage prompt (multilingual, short emotional beats)",
     ["triage", "examples", "few-shot", "fewshot", "multilingual", "bilingual",
      "russian", "code-mixed", "mixed", "short", "emotional", "narrative"]),
    ("Add explicit JSON schema / required fields to extract prompt",
     ["extract prompt", "schema example", "explicit", "required", "output schema",
      "skeleton", "template", "json schema"]),
    ("Cheaper model for triage / Opus overkill — hybrid with Sonnet/Haiku/Qwen",
     ["opus", "overkill", "sonnet", "haiku", "qwen", "hybrid", "cost", "cheap",
      "distill", "downgrade", "expensive"]),
    ("Resolver thresholds (0.98/0.7) and first-token 0.85 boost too permissive for common names",
     ["resolver", "threshold", "gate", "0.98", "0.7", "0.85", "first-token",
      "first token", "permissive", "noisy", "spam", "soft match", "similarity"]),
    ("Scorer v1.0 has no calibration / cross-observation normalization / decay",
     ["scorer", "calibration", "normalization", "normalize", "decay", "drift",
      "baseline", "inconsistent"]),
    ("No observability / no extraction benchmark / no ground truth",
     ["observability", "metric", "metrics", "telemetry", "logging", "log",
      "monitoring", "benchmark", "ground truth", "golden", "evaluation",
      "precision", "recall"]),
    ("Concurrency / race conditions on entities table — double-inserts",
     ["race", "concurrency", "concurrent", "lock", "locking", "parallel",
      "transaction", "commit"]),
    ("Stale graph context — existing_entities loaded per-observation won't scale",
     ["existing_entities", "graph context", "context", "stale", "loaded",
      "load", "scale", "pre-retrieve", "top-k", "pagination", "all entities"]),
]


def _norm_item(s):
    return re.sub(r"\s+", " ", s.strip())


def _merge_and_dedup(items, kind="bug"):
    items = [_norm_item(s) for s in items if s and s.strip()]
    # Precompute tokens
    tokenized = [(s, _tokens(s)) for s in items]
    theme_sets = [(name, set(kws)) for name, kws in _THEMES]
    bucketed: dict[str, list[str]] = {name: [] for name, _ in _THEMES}
    unbucketed: list[str] = []
    for orig, toks in tokenized:
        best_name = None
        best_hits = 0
        low = orig.lower()
        for name, kws in theme_sets:
            hits = sum(1 for kw in kws if (" " in kw and kw in low) or (" " not in kw and kw in toks))
            if hits > best_hits:
                best_hits = hits
                best_name = name
        if best_name and best_hits >= 2:
            bucketed[best_name].append(orig)
        else:
            unbucketed.append(orig)
    # Also cluster unbucketed by Jaccard
    jacc_clusters: list[dict] = []
    for orig in unbucketed:
        toks = _tokens(orig)
        best_j = 0.0
        best_c = None
        for c in jacc_clusters:
            j = _jaccard(toks, c["toks"])
            if j > best_j:
                best_j = j
                best_c = c
        if best_c and best_j >= 0.35:
            best_c["members"].append(orig)
            best_c["toks"] |= toks
        else:
            jacc_clusters.append({"members": [orig], "toks": toks})
    results = []
    for name, members in bucketed.items():
        if members:
            results.append((name, len(members), members))
    for c in jacc_clusters:
        if c["members"]:
            results.append((c["members"][0], len(c["members"]), c["members"]))
    results.sort(key=lambda r: -r[1])
    return results


results = {}
for jid, _label in JUDGE_ORDER:
    p = RAW / f"{jid}.json"
    if p.exists():
        results[jid] = json.loads(p.read_text(encoding="utf-8"))

verdicts_ok = [r["verdict"] for r in results.values() if r.get("verdict")]

score_rows = []
for key in SCORE_KEYS:
    vals, avg = _collect_score(verdicts_ok, key)
    score_rows.append((key, avg, len(vals), vals))

overall_vals = []
for v in verdicts_ok:
    n = _pick_int(v.get("overall_score"))
    if n is not None and 1 <= n <= 10:
        overall_vals.append(n)
overall_avg = sum(overall_vals) / len(overall_vals) if overall_vals else None

verdict_counts = {"ship": 0, "rework": 0, "rethink": 0, "other": 0}
for v in verdicts_ok:
    vd = (v.get("verdict") or "").strip().lower()
    if vd in verdict_counts:
        verdict_counts[vd] += 1
    else:
        verdict_counts["other"] += 1

all_bugs = []
all_improvs = []
for v in verdicts_ok:
    for item in (v.get("top_bugs") or []):
        if isinstance(item, str):
            all_bugs.append(item)
        elif isinstance(item, dict):
            t = item.get("bug") or item.get("text") or item.get("title")
            if isinstance(t, str):
                all_bugs.append(t)
    for item in (v.get("top_prompt_improvements") or v.get("top_improvements") or []):
        if isinstance(item, str):
            all_improvs.append(item)
        elif isinstance(item, dict):
            t = item.get("improvement") or item.get("text") or item.get("title")
            if isinstance(t, str):
                all_improvs.append(t)

merged_bugs = _merge_and_dedup(all_bugs, kind="bug")
merged_improvs = _merge_and_dedup(all_improvs, kind="improv")

summary = {
    "run_dir": RUN_DIR.name,
    "judges_total": len(JUDGE_ORDER),
    "judges_ok": len(verdicts_ok),
    "judges_error": sum(1 for r in results.values() if r.get("error")),
    "judges_parse_failed": sum(1 for r in results.values()
                                if not r.get("error") and r.get("verdict") is None),
    "scores": {k: {"avg": avg, "n": n, "values": vals} for k, avg, n, vals in score_rows},
    "overall_score_avg": overall_avg,
    "overall_score_values": overall_vals,
    "verdict_distribution": verdict_counts,
    "top_bugs_merged": [{"theme": t, "count": c, "members": m} for t, c, m in merged_bugs],
    "top_prompt_improvements_merged": [{"theme": t, "count": c, "members": m} for t, c, m in merged_improvs],
    "per_judge": [
        {
            "id": jid,
            "label": label,
            "model": results.get(jid, {}).get("model"),
            "elapsed_s": results.get(jid, {}).get("elapsed_s"),
            "error": results.get(jid, {}).get("error"),
            "parse_failed": (results.get(jid, {}).get("verdict") is None and not results.get(jid, {}).get("error")),
            "overall_score": (results.get(jid, {}).get("verdict") or {}).get("overall_score"),
            "verdict": (results.get(jid, {}).get("verdict") or {}).get("verdict"),
            "scores": (results.get(jid, {}).get("verdict") or {}).get("scores"),
        }
        for jid, label in JUDGE_ORDER
        if jid in results
    ],
}

(RUN_DIR / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ---- report.md ----
lines = []
lines.append("# Pulse Extraction Pipeline — Design Review Report")
lines.append("")
lines.append(f"Run: `{RUN_DIR.name}`")
lines.append("")
lines.append(f"- Judges total: {len(JUDGE_ORDER)}")
lines.append(f"- OK: {len(verdicts_ok)}")
lines.append(f"- Parse-failed: {summary['judges_parse_failed']}")
lines.append(f"- Errored: {summary['judges_error']}")
lines.append("")

lines.append("## Average scores (1–10)")
lines.append("")
lines.append("| # | Category | Avg | n | Values |")
lines.append("|---:|---|---:|---:|---|")
for key, avg, n, vals in score_rows:
    num = key.split("_", 1)[0]
    label = key.split("_", 1)[1].replace("_", " ")
    avg_s = f"{avg:.2f}" if avg is not None else "—"
    lines.append(f"| {num} | {label} | {avg_s} | {n} | {vals} |")
overall_s = f"{overall_avg:.2f}" if overall_avg is not None else "—"
lines.append(f"| | **overall_score** | **{overall_s}** | {len(overall_vals)} | {overall_vals} |")
lines.append("")

lines.append("## Verdict distribution")
lines.append("")
total_votes = sum(verdict_counts.values())
for k in ["ship", "rework", "rethink", "other"]:
    c = verdict_counts[k]
    pct = (100 * c / total_votes) if total_votes else 0
    lines.append(f"- **{k}**: {c} ({pct:.0f}%)")
lines.append("")

lines.append("## Per-judge summary")
lines.append("")
lines.append("| Judge | Model | Overall | Verdict | Elapsed | Status |")
lines.append("|---|---|---:|---|---:|---|")
for jid, label in JUDGE_ORDER:
    if jid not in results:
        continue
    r = results[jid]
    v = r.get("verdict") or {}
    ov = v.get("overall_score")
    vd = v.get("verdict")
    status = "ok" if v else ("parse_failed" if not r.get("error") else "error")
    ov_s = str(ov) if ov is not None else "—"
    vd_s = vd if vd else (r.get("error") or "—")
    lines.append(
        f"| {label} | `{r.get('model','?')}` | {ov_s} | {vd_s} | {r.get('elapsed_s','?')}s | {status} |"
    )
lines.append("")

lines.append(f"## Top bugs / risks (themed, across {len(verdicts_ok)} judges)")
lines.append("")
if merged_bugs:
    for theme, count, members in merged_bugs[:15]:
        lines.append(f"### [{count} judges] {theme}")
        for m in members[:5]:
            lines.append(f"  - _{m}_")
        if len(members) > 5:
            lines.append(f"  - _…and {len(members)-5} more_")
        lines.append("")
else:
    lines.append("_(no bugs extracted)_")
lines.append("")

lines.append("## Top prompt / pipeline improvements (themed)")
lines.append("")
if merged_improvs:
    for theme, count, members in merged_improvs[:15]:
        lines.append(f"### [{count} judges] {theme}")
        for m in members[:5]:
            lines.append(f"  - _{m}_")
        if len(members) > 5:
            lines.append(f"  - _…and {len(members)-5} more_")
        lines.append("")
else:
    lines.append("_(no improvements extracted)_")
lines.append("")

# Full texts for key judges
for key_id in ["gpt54pro", "opus"]:
    r = results.get(key_id)
    if not r:
        continue
    label = dict(JUDGE_ORDER).get(key_id, key_id)
    lines.append(f"## Full response — {label}")
    lines.append("")
    raw_md_path = RAW / f"{key_id}.md"
    raw_text = raw_md_path.read_text(encoding="utf-8") if raw_md_path.exists() else ""
    if r.get("error"):
        lines.append(f"_Error: {r['error']}_")
    else:
        lines.append(raw_text or "_(empty)_")
    lines.append("")

lines.append("## Artifacts")
lines.append("")
lines.append(f"- `raw/<judge>.md` — full response per judge")
lines.append(f"- `raw/<judge>.json` — extracted verdict JSON")
lines.append(f"- `summary.json` — machine-readable aggregate")
lines.append("")

(RUN_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

print(f"wrote {RUN_DIR}/report.md")
print(f"wrote {RUN_DIR}/summary.json")
print(f"ok={len(verdicts_ok)}/{len(JUDGE_ORDER)}  overall_avg={overall_s}  verdicts={verdict_counts}")
