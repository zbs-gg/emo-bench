#!/usr/bin/env python3
"""
Memory retrieval bench — 6-way comparison.

Systems:
  A = MemPalace (VDS, local all-MiniLM-L6-v2)
  B = sqlite-vec (VDS, OpenAI text-embedding-3-large via OpenClaw memory)
  C = SillyTavern Vector Storage + transformers (local Mac, jina-embeddings-v2-base-en)
  D = SillyTavern Vector Storage + openai (local Mac, text-embedding-3-large)
  E = Arkhon Memory (local Mac, sentence-transformers/all-MiniLM-L6-v2 + FAISS)
  F = emogie Advanced Memory (local Mac, all-MiniLM-L6-v2 + ChromaDB)

Uses the same 10 queries as bench-memory-retrieval.py.
Opus judge scores each system 0-10 on relevance/specificity/actionability,
then picks the best system per query.

Run: python3 bench-memory-6way.py
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib import request as urlreq
from urllib.error import HTTPError

import anthropic

API_KEY = os.environ.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("No claude_api_key or ANTHROPIC_API_KEY in env")

client = anthropic.Anthropic(api_key=API_KEY)
JUDGE_MODEL = "claude-opus-4-6"
VDS = "openclaw@152.42.186.145"
ST_URL = "http://localhost:8000"
ARKHON_URL = "http://localhost:9000"
EMOGIE_URL = "http://localhost:5125"
USER_ID = "bench"
CHAR_NAME = "bench"

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
        ["ssh", VDS, cmd], capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout


def post_json(url: str, payload: dict, timeout: int = 60) -> tuple[int, dict | list | None]:
    data = json.dumps(payload).encode("utf-8")
    req = urlreq.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlreq.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, None
    except Exception:
        return -1, None


def query_mempalace(q: str, k: int = 3) -> str:
    q_esc = q.replace("'", "'\\''")
    cmd = f"~/mempalace-env/bin/mempalace --palace ~/persistent/mempalace search '{q_esc}' --results {k} 2>/dev/null"
    return run_ssh(cmd, timeout=30).strip()


def query_sqlite_vec(q: str, k: int = 3) -> str:
    q_esc = q.replace("'", "'\\''")
    cmd = f"openclaw memory search '{q_esc}' --max-results {k} --min-score 0 2>/dev/null"
    out = run_ssh(cmd, timeout=90)
    lines = [ln for ln in out.splitlines() if "plugin" not in ln.lower() and "Config warnings" not in ln]
    return "\n".join(lines).strip()


def query_st(q: str, collection: str, source: str, k: int = 3, model: str = "") -> str:
    payload = {
        "collectionId": collection,
        "source": source,
        "searchText": q,
        "topK": k,
        "threshold": 0.0,
    }
    if model:
        payload["model"] = model
    data = json.dumps(payload).encode()
    req = urlreq.Request(
        f"{ST_URL}/api/vector/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlreq.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
    except HTTPError as e:
        return f"(HTTP {e.code}: {e.read()[:100].decode('utf8', errors='replace')})"
    except Exception as e:
        return f"(error: {e})"
    md = d.get("metadata", [])
    if not md:
        return "(no results)"
    parts = []
    for i, m in enumerate(md, 1):
        text = m.get("text", "").strip()
        parts.append(f"Result {i}:\n{text}")
    return "\n\n".join(parts)


def query_st_transformers(q: str, k: int = 3) -> str:
    return query_st(q, "bench-tr", "transformers", k)


def query_st_openai(q: str, k: int = 3) -> str:
    return query_st(q, "bench-oa", "openai", k, model="text-embedding-3-large")


def query_arkhon(q: str, k: int = 3) -> str:
    status, data = post_json(
        f"{ARKHON_URL}/memories/recall",
        {"user_id": USER_ID, "char_name": CHAR_NAME, "query": q, "top_k": k},
        timeout=30,
    )
    if status != 200 or not isinstance(data, list):
        return f"(HTTP {status})"
    if not data:
        return "(no results)"
    parts = []
    for i, m in enumerate(data[:k], 1):
        text = (m.get("text") or "").strip()
        parts.append(f"Result {i}:\n{text}")
    return "\n\n".join(parts)


def query_emogie(q: str, k: int = 3) -> str:
    status, data = post_json(
        f"{EMOGIE_URL}/memory/query",
        {"query": q, "k": k, "min_score": 0.0},
        timeout=30,
    )
    if status != 200 or not isinstance(data, dict):
        return f"(HTTP {status})"
    results = data.get("results", [])
    if not results:
        return "(no results)"
    parts = []
    for i, m in enumerate(results[:k], 1):
        text = (m.get("text") or "").strip()
        # emogie wraps as "[name] (role): content" — keep as is, judge can parse
        parts.append(f"Result {i}:\n{text}")
    return "\n\n".join(parts)


JUDGE_SYSTEM = """You judge memory retrieval systems.
You will see one query and six retrieved result sets (A, B, C, D, E, F).
Rate each system 0-10 on:
- relevance: does the retrieval actually address the query?
- specificity: concrete facts, dates, names vs vague paragraphs
- actionability: can an AI companion use this to answer Nik better?

Then pick a winner (A, B, C, D, E, F, or tie). Tie only if multiple systems are clearly equivalent at the top.

Respond with JSON only:
{"A_rel":N,"A_spec":N,"A_act":N,"B_rel":N,"B_spec":N,"B_act":N,"C_rel":N,"C_spec":N,"C_act":N,"D_rel":N,"D_spec":N,"D_act":N,"E_rel":N,"E_spec":N,"E_act":N,"F_rel":N,"F_spec":N,"F_act":N,"winner":"A|B|C|D|E|F|tie","note":"one short sentence"}
"""


def judge(q: str, a: str, b: str, c: str, d: str, e: str, f: str) -> dict:
    cap = 1800
    user = f"""Query: "{q}"

=== SYSTEM A (MemPalace, local MiniLM embeddings) ===
{(a or '(empty)')[:cap]}

=== SYSTEM B (sqlite-vec via OpenClaw, OpenAI text-embedding-3-large) ===
{(b or '(empty)')[:cap]}

=== SYSTEM C (SillyTavern Vector Storage, local transformers jina-v2) ===
{(c or '(empty)')[:cap]}

=== SYSTEM D (SillyTavern Vector Storage, OpenAI text-embedding-3-large) ===
{(d or '(empty)')[:cap]}

=== SYSTEM E (Arkhon Memory, local all-MiniLM-L6-v2 + FAISS) ===
{(e or '(empty)')[:cap]}

=== SYSTEM F (emogie Advanced Memory, local all-MiniLM-L6-v2 + ChromaDB) ===
{(f or '(empty)')[:cap]}
"""
    try:
        r = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=800,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = r.content[0].text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"error": "no_json", "raw": text[:200]}
    except Exception as exc:
        return {"error": str(exc)[:200]}


def avg(results: list, system: str, key: str) -> float:
    vals = [r["v"].get(f"{system}_{key}", 0) for r in results if "error" not in r["v"]]
    return sum(vals) / len(vals) if vals else 0.0


SYSTEMS = ["A", "B", "C", "D", "E", "F"]
NAMES = {
    "A": "MemPalace",
    "B": "sqlite-vec",
    "C": "ST-transformers",
    "D": "ST-openai",
    "E": "Arkhon",
    "F": "emogie",
}


def main():
    print(f"=== Memory retrieval bench (6-way) ===")
    print(f"A=MemPalace B=sqlite-vec C=ST-tr D=ST-oa E=Arkhon F=emogie")
    print(f"Queries: {len(QUERIES)} | Judge: {JUDGE_MODEL}\n")

    results = []
    t0 = time.time()
    for i, q in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] {q[:60]}", flush=True)
        print("  A mempalace...", end=" ", flush=True)
        a = query_mempalace(q)
        print(f"{len(a)}ch", end=" | ")
        print("B sqlite-vec...", end=" ", flush=True)
        b = query_sqlite_vec(q)
        print(f"{len(b)}ch", end=" | ")
        print("C ST-tr...", end=" ", flush=True)
        c = query_st_transformers(q)
        print(f"{len(c)}ch", end=" | ")
        print("D ST-oa...", end=" ", flush=True)
        d = query_st_openai(q)
        print(f"{len(d)}ch", end=" | ")
        print("E arkhon...", end=" ", flush=True)
        e = query_arkhon(q)
        print(f"{len(e)}ch", end=" | ")
        print("F emogie...", end=" ", flush=True)
        f = query_emogie(q)
        print(f"{len(f)}ch", end=" | ")
        print("judge...", end=" ", flush=True)
        v = judge(q, a, b, c, d, e, f)
        winner = v.get("winner", "?")
        note = (v.get("note", "") or "")[:60]
        print(f"→ {winner}: {note}")
        results.append({"q": q, "a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "v": v})

    elapsed = time.time() - t0

    # Aggregate
    n = len(results)
    wins = {s: sum(1 for r in results if r["v"].get("winner") == s) for s in SYSTEMS}
    ties = sum(1 for r in results if r["v"].get("winner") == "tie")

    scores = {}
    for s in SYSTEMS:
        rel = avg(results, s, "rel")
        spec = avg(results, s, "spec")
        act = avg(results, s, "act")
        scores[s] = {"rel": rel, "spec": spec, "act": act, "total": rel + spec + act}

    print(f"\n=== RESULTS ({n} queries, {elapsed:.1f}s) ===\n")
    print(f"Wins: " + "  ".join(f"{NAMES[s]}={wins[s]}" for s in SYSTEMS) + f"  ties={ties}")
    print()
    print(f"{'System':<18} {'Rel':>8} {'Spec':>8} {'Act':>8} {'TOTAL/30':>11}")
    for s in SYSTEMS:
        name = NAMES[s]
        sc = scores[s]
        print(f"{name:<18} {sc['rel']:>8.2f} {sc['spec']:>8.2f} {sc['act']:>8.2f} {sc['total']:>11.2f}")
    print()

    # Write report
    ts = time.strftime("%Y%m%d-%H%M")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"memory-6way-{ts}.json"
    out_md = out_dir / f"memory-6way-{ts}.md"

    out_json.write_text(json.dumps({
        "queries": QUERIES,
        "results": results,
        "summary": {
            "n": n, "wins": wins, "ties": ties,
            "scores": scores,
            "names": NAMES,
            "elapsed_s": elapsed,
        },
    }, ensure_ascii=False, indent=2))

    lines = [
        f"# Memory Retrieval Bench — 6-way",
        f"",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')} | **Judge**: {JUDGE_MODEL}",
        f"",
        f"| Letter | System | Embeddings | Runs on |",
        f"|---|---|---|---|",
        f"| A | MemPalace | all-MiniLM-L6-v2 (local) | VDS |",
        f"| B | sqlite-vec (OpenClaw) | text-embedding-3-large | VDS |",
        f"| C | SillyTavern Vector Storage | jina-embeddings-v2-base-en (local) | Mac |",
        f"| D | SillyTavern Vector Storage | text-embedding-3-large | Mac |",
        f"| E | Arkhon Memory | all-MiniLM-L6-v2 (local) + FAISS | Mac :9000 |",
        f"| F | emogie Advanced Memory | all-MiniLM-L6-v2 (local) + ChromaDB | Mac :5125 |",
        f"",
        f"Same corpus (`/tmp/bench-corpus` — 298 markdown files, ~2000 chunks ~500 chars each) ingested into C, D, E, F; A and B query pre-existing indexes.",
        f"",
        f"## Summary",
        f"",
        f"| System | Wins | Relevance | Specificity | Actionability | **Total (0-30)** |",
        f"|---|---|---|---|---|---|",
    ]
    for s in SYSTEMS:
        name = NAMES[s]
        sc = scores[s]
        lines.append(f"| {name} | **{wins[s]}** | {sc['rel']:.2f} | {sc['spec']:.2f} | {sc['act']:.2f} | **{sc['total']:.2f}** |")
    lines.append(f"")
    lines.append(f"Ties: {ties}")
    lines.append(f"")
    lines.append(f"## Per-query")
    lines.append(f"")
    for i, r in enumerate(results, 1):
        v = r["v"]
        lines.append(f"### {i}. {r['q']}")
        lines.append(f"**Winner**: {v.get('winner','?')} — {v.get('note','')}")
        lines.append("")
        lines.append("| | Rel | Spec | Act |")
        lines.append("|---|---|---|---|")
        for s in SYSTEMS:
            lines.append(f"| {NAMES[s]} | {v.get(f'{s}_rel','?')} | {v.get(f'{s}_spec','?')} | {v.get(f'{s}_act','?')} |")
        lines.append("")
        for letter, key, label in [
            ("A", "a", "MemPalace"),
            ("B", "b", "sqlite-vec"),
            ("C", "c", "ST-transformers"),
            ("D", "d", "ST-openai"),
            ("E", "e", "Arkhon"),
            ("F", "f", "emogie"),
        ]:
            lines.append(f"<details><summary>{label} output</summary>")
            lines.append("")
            lines.append("```")
            lines.append((r[key] or "(empty)")[:1800])
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"Report: {out_md}")
    print(f"JSON:   {out_json}")


if __name__ == "__main__":
    main()
