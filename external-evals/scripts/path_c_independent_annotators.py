#!/usr/bin/env python3
"""
Path C — Independent Annotator Pilot
=====================================
Addresses peer-review concern: ideal_top_3_event_ids were authored by the
project owner with no inter-annotator agreement check.

Strategy: prompt 3 independent LLM families (Claude, OpenAI, Grok) blind to
Pulse v3 and to the original labels. Measure pairwise Jaccard and agreement
with author labels across 5 stateful probe pairs (one A-variant per pair).
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _load_env():
    env_path = Path(__file__).parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4.1"          # gpt-5.4 alias — use stable name

# xAI Grok key — set XAI_API_KEY in your .env (see .env.example)
GROK_KEY   = os.environ.get("XAI_API_KEY", "")
GROK_API   = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-3-fast"          # lightweight; grok-4 if available

CLAUDE_MODEL = "claude-sonnet-4-6"  # via `claude -p`; subscription = $0

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an independent memory-annotation expert. "
    "You know nothing about any specific retrieval architecture or system. "
    "Your task: given a set of longitudinal user memory events, identify which 3 events "
    "would be most valuable for a thoughtful memory companion to surface in response to a "
    "specific probe (user query + emotional state). "
    "Return ONLY a JSON array of exactly 3 integers (event IDs), e.g. [4, 17, 42]. "
    "No explanation, no markdown, no surrounding text."
)

def _build_user_prompt(events: list[dict], test: dict) -> str:
    lines = ["--- EVENTS ---"]
    for e in events:
        lines.append(f"ID {e['id']}: {e['text']}")
    lines.append("")
    lines.append("--- PROBE ---")
    lines.append(f"user_query: {test['user_query']}")
    lines.append(f"user_state: {json.dumps(test['user_state'], ensure_ascii=False)}")
    lines.append("")
    lines.append(
        "Which 3 event IDs would best serve as ideal memory retrieval for this probe? "
        "Return a JSON array of exactly 3 integers."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Annotator: Claude Sonnet via `claude -p`
# ---------------------------------------------------------------------------

def annotate_claude(events: list[dict], test: dict) -> list[int]:
    prompt = _build_user_prompt(events, test)
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
    result = subprocess.run(
        ["claude", "-p", "--model", CLAUDE_MODEL, full_prompt],
        capture_output=True, text=True, timeout=120
    )
    raw = result.stdout.strip()
    return _parse_ids(raw)


# ---------------------------------------------------------------------------
# Annotator: OpenAI GPT via API
# ---------------------------------------------------------------------------

def annotate_openai(events: list[dict], test: dict) -> list[int]:
    import urllib.request
    prompt = _build_user_prompt(events, test)
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 60,
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    raw = data["choices"][0]["message"]["content"].strip()
    return _parse_ids(raw)


# ---------------------------------------------------------------------------
# Annotator: xAI Grok via API
# ---------------------------------------------------------------------------

def annotate_grok(events: list[dict], test: dict) -> list[int]:
    import urllib.request
    prompt = _build_user_prompt(events, test)
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 60,
        "temperature": 0,
    }
    req = urllib.request.Request(
        GROK_API,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROK_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    raw = data["choices"][0]["message"]["content"].strip()
    return _parse_ids(raw)


# ---------------------------------------------------------------------------
# Parse JSON ids from LLM output
# ---------------------------------------------------------------------------

def _parse_ids(raw: str) -> list[int]:
    # Try to find JSON array anywhere in the output
    m = re.search(r"\[[\d\s,]+\]", raw)
    if m:
        ids = json.loads(m.group())
        return [int(x) for x in ids[:3]]
    # fallback: extract all integers
    nums = [int(x) for x in re.findall(r"\d+", raw)]
    if len(nums) >= 3:
        return nums[:3]
    raise ValueError(f"Could not parse 3 IDs from: {raw!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def mean_jaccard_pair(all_annotations: dict[str, dict[str, list[int]]],
                       a_key: str, b_key: str, probe_ids: list[str]) -> float:
    scores = []
    for pid in probe_ids:
        a_ids = all_annotations[a_key].get(pid)
        b_ids = all_annotations[b_key].get(pid)
        if a_ids and b_ids:
            scores.append(jaccard(a_ids, b_ids))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def mean_jaccard_vs_author(annotator_ids: dict[str, list[int]],
                            author_ids: dict[str, list[int]],
                            probe_ids: list[str]) -> float:
    scores = []
    for pid in probe_ids:
        a_ids = annotator_ids.get(pid)
        auth_ids = author_ids.get(pid)
        if a_ids and auth_ids:
            scores.append(jaccard(a_ids, auth_ids))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print prompts only, skip API calls")
    args = ap.parse_args()

    with open(args.corpus) as f:
        corpus = json.load(f)

    events: list[dict] = corpus["events"]
    tests: list[dict]  = corpus["tests"]

    # Select one A-variant per pair (5 pairs total: P1-P5)
    # Pairs: P1=T6/T7, P2=T8/T9, P3=T10/T11, P4=T12/T13, P5=T14/T15
    target_ids = {"T6", "T8", "T10", "T12", "T14"}
    probe_tests = [t for t in tests if t["id"] in target_ids]
    probe_tests.sort(key=lambda t: t["id"])

    print(f"Selected {len(probe_tests)} probes: {[t['id'] for t in probe_tests]}")

    author_ideal: dict[str, list[int]] = {
        t["id"]: t["ideal_top_3_event_ids"] for t in probe_tests
    }

    annotators = {
        "claude-sonnet-4.6": annotate_claude,
        "openai-gpt-4.1":    annotate_openai,
        "xai-grok":          annotate_grok,
    }
    annotator_ideal: dict[str, dict[str, Any]] = {a: {} for a in annotators}
    failures: list[str] = []

    for test in probe_tests:
        tid = test["id"]
        print(f"\n=== Probe {tid}: {test['name']} ===")
        print(f"  author ideal: {author_ideal[tid]}")

        for ann_name, ann_fn in annotators.items():
            if args.dry_run:
                print(f"  [{ann_name}] DRY RUN — skipping")
                annotator_ideal[ann_name][tid] = None
                continue
            try:
                ids = ann_fn(events, test)
                print(f"  [{ann_name}] → {ids}")
                annotator_ideal[ann_name][tid] = ids
            except Exception as e:
                msg = f"{ann_name}/{tid}: {e}"
                print(f"  [{ann_name}] FAILED: {e}")
                failures.append(msg)
                annotator_ideal[ann_name][tid] = None

    probe_ids = [t["id"] for t in probe_tests]
    ann_keys  = list(annotators.keys())

    # Pairwise Jaccard between annotators
    pairwise: dict[str, float] = {}
    for i in range(len(ann_keys)):
        for j in range(i + 1, len(ann_keys)):
            a, b = ann_keys[i], ann_keys[j]
            label = f"{a.split('-')[0]}-vs-{b.split('-')[0]}"
            pairwise[label] = mean_jaccard_pair(annotator_ideal, a, b, probe_ids)

    # Agreement with author
    agreement_with_author: dict[str, float] = {}
    for ann_name in ann_keys:
        ann_clean = {k: v for k, v in annotator_ideal[ann_name].items() if v}
        agreement_with_author[ann_name] = mean_jaccard_vs_author(
            ann_clean, author_ideal, probe_ids
        )

    # Interpretation
    avg_pairwise = (
        sum(pairwise.values()) / len(pairwise) if pairwise else 0.0
    )
    avg_author_agreement = (
        sum(agreement_with_author.values()) / len(agreement_with_author)
        if agreement_with_author else 0.0
    )

    if avg_author_agreement >= 0.5:
        verdict = "supports"
    elif avg_author_agreement >= 0.25:
        verdict = "partially supports"
    else:
        verdict = "weakens"

    interpretation = (
        f"Independent LLMs converge with author labels at mean Jaccard "
        f"{avg_author_agreement:.2f} (pairwise inter-LLM: {avg_pairwise:.2f}). "
        f"This {verdict} the author-authored framing as cross-family consistent."
    )
    if failures:
        interpretation += f" {len(failures)} call(s) failed: see failures field."

    result = {
        "probes":    probe_ids,
        "annotators": ann_keys,
        "author_ideal": author_ideal,
        "annotator_ideal": annotator_ideal,
        "pairwise_jaccard": pairwise,
        "agreement_with_author": agreement_with_author,
        "interpretation": interpretation,
        "failures": failures,
        "meta": {
            "corpus": str(args.corpus),
            "run_at": datetime.utcnow().isoformat() + "Z",
            "openai_model": OPENAI_MODEL,
            "grok_model": GROK_MODEL,
            "claude_model": CLAUDE_MODEL,
        }
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n=== Results written to {out_path} ===")
    print(f"Pairwise Jaccard: {pairwise}")
    print(f"Agreement with author: {agreement_with_author}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
