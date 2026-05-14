#!/usr/bin/env python3
"""
Memory retrieval bench — MemPalace vs sqlite-vec (OpenClaw built-in).

Fires identical queries at both systems on VDS via ssh, asks Opus to judge
which retrieval is more relevant/specific/actionable. Reports winner per
query and aggregate scores.

Run: python3 bench-memory-retrieval.py
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import anthropic

API_KEY = os.environ.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("No claude_api_key or ANTHROPIC_API_KEY in env")

client = anthropic.Anthropic(api_key=API_KEY)
JUDGE_MODEL = "claude-opus-4-6"
VDS = "openclaw@152.42.186.145"

QUERIES = [
    "Garden companion memory architecture",
    "Mila project brand voice and aesthetic",
    "watchdog timeout root cause in gateway",
    "Nik emotional IFS parts and anchors",
    "Sonya book remaster design decisions",
    "SOUL.md Elle identity and autonomy",
    "Freeman TGE roadmap status",
    "heartbeat daily notes routine",
    "workspace systemd services on VDS",
    "Mar 15 diary trust moment",
]


def run_ssh(cmd: str, timeout: int = 60) -> str:
    r = subprocess.run(
        ["ssh", VDS, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.stdout


def query_mempalace(q: str, k: int = 3) -> str:
    # mempalace search has no config warnings, pure output
    q_esc = q.replace("'", "'\\''")
    cmd = f"~/mempalace-env/bin/mempalace --palace ~/persistent/mempalace search '{q_esc}' --results {k} 2>/dev/null"
    return run_ssh(cmd, timeout=30).strip()


def query_sqlite_vec(q: str, k: int = 3) -> str:
    # openclaw memory search writes config warnings to stderr — drop them
    # Use --min-score 0 because default filters out almost everything
    q_esc = q.replace("'", "'\\''")
    cmd = f"openclaw memory search '{q_esc}' --max-results {k} --min-score 0 2>/dev/null"
    out = run_ssh(cmd, timeout=90)
    # Strip the plugin warning banner if it leaks
    lines = [ln for ln in out.splitlines() if "plugin" not in ln.lower() and "Config warnings" not in ln]
    return "\n".join(lines).strip()


JUDGE_SYSTEM = """You judge memory retrieval systems.
You will see one query and two retrieved result sets (A and B).
Rate each system 0-10 on:
- relevance: does the retrieval actually address the query?
- specificity: concrete facts, dates, names vs vague paragraphs
- actionability: can an AI companion use this to answer Nik better?

Then pick a winner (A, B, or tie).

Respond with JSON only:
{"A_relevance":N,"A_specificity":N,"A_actionability":N,"B_relevance":N,"B_specificity":N,"B_actionability":N,"winner":"A|B|tie","note":"one short sentence"}
"""


def judge(q: str, a: str, b: str) -> dict:
    # Cap each to 3000 chars to keep judge prompt reasonable
    a_trimmed = (a or "(empty)")[:3000]
    b_trimmed = (b or "(empty)")[:3000]
    user = f"""Query: "{q}"

=== SYSTEM A (MemPalace, local MiniLM embeddings) ===
{a_trimmed}

=== SYSTEM B (sqlite-vec, OpenAI text-embedding-3-large) ===
{b_trimmed}
"""
    try:
        r = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=400,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = r.content[0].text
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"error": "no_json", "raw": text[:200]}
    except Exception as e:
        return {"error": str(e)[:200]}


def main():
    print(f"=== Memory retrieval bench: MemPalace vs sqlite-vec ===")
    print(f"Queries: {len(QUERIES)} | Judge: {JUDGE_MODEL}\n")

    results = []
    t0 = time.time()
    for i, q in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] {q[:70]}")
        print("  mempalace...", end=" ", flush=True)
        mp = query_mempalace(q)
        print(f"{len(mp)} chars", end=" | ")
        print("sqlite-vec...", end=" ", flush=True)
        sv = query_sqlite_vec(q)
        print(f"{len(sv)} chars", end=" | ")
        print("judge...", end=" ", flush=True)
        v = judge(q, mp, sv)
        winner = v.get("winner", "?")
        note = v.get("note", "")[:60]
        print(f"→ {winner}: {note}")
        results.append({"q": q, "mp": mp, "sv": sv, "v": v})

    elapsed = time.time() - t0

    # Aggregate
    n = len(results)
    a_wins = sum(1 for r in results if r["v"].get("winner") == "A")
    b_wins = sum(1 for r in results if r["v"].get("winner") == "B")
    ties = n - a_wins - b_wins

    def avg(sys: str, key: str) -> float:
        vals = [r["v"].get(f"{sys}_{key}", 0) for r in results if "error" not in r["v"]]
        return sum(vals) / len(vals) if vals else 0.0

    a_rel, a_spec, a_act = avg("A", "relevance"), avg("A", "specificity"), avg("A", "actionability")
    b_rel, b_spec, b_act = avg("B", "relevance"), avg("B", "specificity"), avg("B", "actionability")
    a_total = a_rel + a_spec + a_act
    b_total = b_rel + b_spec + b_act

    print(f"\n=== RESULTS ({n} queries, {elapsed:.1f}s) ===\n")
    print(f"MemPalace wins: {a_wins} | sqlite-vec wins: {b_wins} | ties: {ties}")
    print()
    print(f"{'Metric':<16} {'MemPalace':>12} {'sqlite-vec':>12}")
    print(f"{'Relevance':<16} {a_rel:>12.2f} {b_rel:>12.2f}")
    print(f"{'Specificity':<16} {a_spec:>12.2f} {b_spec:>12.2f}")
    print(f"{'Actionability':<16} {a_act:>12.2f} {b_act:>12.2f}")
    print(f"{'TOTAL (of 30)':<16} {a_total:>12.2f} {b_total:>12.2f}")
    print()

    # Write report
    ts = time.strftime("%Y%m%d-%H%M")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"memory-retrieval-{ts}.json"
    out_md = out_dir / f"memory-retrieval-{ts}.md"

    out_json.write_text(json.dumps({
        "queries": QUERIES,
        "results": results,
        "summary": {
            "n": n, "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
            "A": {"relevance": a_rel, "specificity": a_spec, "actionability": a_act, "total": a_total},
            "B": {"relevance": b_rel, "specificity": b_spec, "actionability": b_act, "total": b_total},
            "elapsed_s": elapsed,
        },
    }, ensure_ascii=False, indent=2))

    lines = [
        f"# Memory Retrieval Bench — MemPalace vs sqlite-vec",
        f"",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')} | **Judge**: {JUDGE_MODEL}",
        f"**A = MemPalace** (local all-MiniLM-L6-v2) | **B = sqlite-vec** (OpenAI text-embedding-3-large)",
        f"**Corpus**: both query VDS palace / memory as-is (already populated)",
        f"",
        f"## Summary",
        f"",
        f"| Metric | MemPalace | sqlite-vec |",
        f"|---|---|---|",
        f"| Wins | **{a_wins}** | **{b_wins}** (ties: {ties}) |",
        f"| Relevance (0-10) | {a_rel:.2f} | {b_rel:.2f} |",
        f"| Specificity (0-10) | {a_spec:.2f} | {b_spec:.2f} |",
        f"| Actionability (0-10) | {a_act:.2f} | {b_act:.2f} |",
        f"| **Total (0-30)** | **{a_total:.2f}** | **{b_total:.2f}** |",
        f"",
        f"## Per-query",
        f"",
    ]
    for i, r in enumerate(results, 1):
        v = r["v"]
        lines.append(f"### {i}. {r['q']}")
        lines.append(f"**Winner**: {v.get('winner','?')} — {v.get('note','')}")
        lines.append(f"**Scores**: A=rel{v.get('A_relevance','?')}/spec{v.get('A_specificity','?')}/act{v.get('A_actionability','?')} | B=rel{v.get('B_relevance','?')}/spec{v.get('B_specificity','?')}/act{v.get('B_actionability','?')}")
        lines.append("")
        lines.append("<details><summary>MemPalace output</summary>")
        lines.append("")
        lines.append("```")
        lines.append(r["mp"][:2000] or "(empty)")
        lines.append("```")
        lines.append("</details>")
        lines.append("")
        lines.append("<details><summary>sqlite-vec output</summary>")
        lines.append("")
        lines.append("```")
        lines.append(r["sv"][:2000] or "(empty)")
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"Report: {out_md}")
    print(f"JSON:   {out_json}")


if __name__ == "__main__":
    main()
