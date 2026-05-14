#!/usr/bin/env python3
"""Retry parse-failed judges with higher output-token budget, then re-aggregate."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib import request as urlreq

import anthropic

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"

# Accept run dir as arg
if len(sys.argv) < 2:
    sys.exit("usage: bench-pulse-retry.py <run-dir-name-under-results>")

RUN_DIR = RESULTS_DIR / sys.argv[1]
if not RUN_DIR.exists():
    sys.exit(f"no such run dir: {RUN_DIR}")
RAW_DIR = RUN_DIR / "raw"

BUNDLE = (BENCH_DIR / "datasets" / "pulse-extraction-design.md").read_text(encoding="utf-8")

API_KEY = os.environ.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY)

JUDGE_SYSTEM = (
    "You are a senior staff engineer reviewing the design of an LLM-powered "
    "information extraction pipeline. Be critical but constructive. Identify "
    "real risks, not theoretical ones. If you recommend changes, be specific "
    "— show code deltas or prompt edits when possible. End with the required "
    "JSON verdict block exactly as specified; do not add trailing prose. "
    "KEEP YOUR ANALYSIS CONCISE: 2-4 sentences per section. The final JSON "
    "block is mandatory and must fit within the response budget."
)

JUDGES_TO_RETRY = {
    "haiku":    {"model": "claude-haiku-4-5-20251001", "provider": "anthropic",   "budget": 16000, "label": "Haiku 4.5"},
    "gpt54pro": {"model": "gpt-5.4-pro",              "provider": "openai_responses", "budget": 32000, "label": "GPT-5.4 Pro"},
}


def _call_anthropic(model: str, user: str, max_tokens: int) -> str:
    r = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in r.content if hasattr(b, "text"))
    if r.stop_reason == "max_tokens":
        text += "\n}"
    return text


def _call_openai_responses(model: str, user: str, max_tokens: int) -> str:
    api_key = os.environ.get("OPENAI_JUDGE_API_KEY", "")
    payload = json.dumps({
        "model": model,
        "instructions": JUDGE_SYSTEM,
        "input": user,
        "max_output_tokens": max_tokens,
    }).encode("utf-8")
    req = urlreq.Request(
        "https://api.openai.com/v1/responses",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlreq.urlopen(req, timeout=900) as r:
        data = json.loads(r.read().decode("utf-8"))
    texts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    if not texts:
        raise RuntimeError(f"no text in response: {json.dumps(data)[:500]}")
    return "\n".join(texts)


def _extract_verdict_json(text: str) -> dict | None:
    stripped = re.sub(r"```(?:json)?\s*", "", text)
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(stripped[start : i + 1])
                    start = -1
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and ("verdict" in obj or "scores" in obj or "overall_score" in obj):
            return obj
    return None


for jid, jinfo in JUDGES_TO_RETRY.items():
    print(f"[retry] {jinfo['label']} (budget={jinfo['budget']})...", flush=True)
    t0 = time.time()
    try:
        if jinfo["provider"] == "anthropic":
            text = _call_anthropic(jinfo["model"], BUNDLE, jinfo["budget"])
        else:
            text = _call_openai_responses(jinfo["model"], BUNDLE, jinfo["budget"])
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{jid}] FAILED after {elapsed:.1f}s: {e}")
        continue
    elapsed = time.time() - t0
    verdict = _extract_verdict_json(text)
    # Overwrite raw md + json
    (RAW_DIR / f"{jid}.md").write_text(text, encoding="utf-8")
    payload = {
        "judge_id": jid,
        "label": jinfo["label"],
        "model": jinfo["model"],
        "provider": jinfo["provider"],
        "elapsed_s": round(elapsed, 1),
        "error": None if text else "empty",
        "parse_failed": verdict is None,
        "verdict": verdict,
    }
    (RAW_DIR / f"{jid}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verdict is None:
        print(f"  [{jid}] still parse_failed ({elapsed:.1f}s, {len(text)} chars)")
    else:
        print(f"  [{jid}] OK ({elapsed:.1f}s): overall={verdict.get('overall_score')} verdict={verdict.get('verdict')}")

print("[retry] done")
