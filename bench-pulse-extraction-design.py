#!/usr/bin/env python3
"""
Pulse Extraction Pipeline — Design Review via 12 LLM judges.

Not a system comparison: a single design bundle (datasets/pulse-extraction-design.md)
is sent to 12 judges (7 companies). Each judge returns a critique + a
machine-readable JSON verdict. We aggregate scores and common themes.

Adapted from bench-empathic-memory.py: reuses JUDGES, NATIVE_APIS, and the
_call_anthropic / _call_native / _call_openai_responses / _call_gemini helpers.

Output: ~/dev/ai/bench/results/pulse-extraction-design-<YYYYMMDD-HHMM>/
  raw/<judge_id>.md    — full response
  raw/<judge_id>.json  — extracted verdict JSON (or {error: ...})
  report.md            — aggregated markdown report
  summary.json         — structured data
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib import request as urlreq
from urllib.error import HTTPError

import anthropic

# --- Paths ---
BENCH_DIR = Path(__file__).parent
DATASETS_DIR = BENCH_DIR / "datasets"
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BUNDLE_FILE = DATASETS_DIR / "pulse-extraction-design.md"
if not BUNDLE_FILE.exists():
    sys.exit(f"Missing design bundle: {BUNDLE_FILE}")

BUNDLE = BUNDLE_FILE.read_text(encoding="utf-8")

# --- API setup ---
API_KEY = os.environ.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("No claude_api_key or ANTHROPIC_API_KEY in env")

client = anthropic.Anthropic(api_key=API_KEY)

# --- Native API endpoints (copied from bench-empathic-memory.py) ---
NATIVE_APIS = {
    "moonshot": {
        "url": "https://api.moonshot.ai/v1/chat/completions",
        "key_env": "MOONSHOT_API_KEY",
    },
    "dashscope": {
        "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "zai": {
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "key_env": "ZAI_API_KEY",
    },
    "xai": {
        "url": "https://api.x.ai/v1/chat/completions",
        "key_env": "XAI_API_KEY",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_JUDGE_API_KEY",
    },
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        "key_env": "GEMINI_API_KEY",
    },
}

# 12 judges from 7 companies (same roster as empathic-memory bench)
JUDGES = [
    {"id": "opus",     "model": "claude-opus-4-6",        "label": "Opus 4.6",       "provider": "anthropic"},
    {"id": "sonnet",   "model": "claude-sonnet-4-6",      "label": "Sonnet 4.6",     "provider": "anthropic"},
    {"id": "haiku",    "model": "claude-haiku-4-5-20251001", "label": "Haiku 4.5",   "provider": "anthropic"},
    {"id": "glm5",     "model": "glm-5.1",                "label": "GLM-5.1",        "provider": "zai"},
    {"id": "deepseek", "model": "deepseek-v3.2",          "label": "DeepSeek V3.2",  "provider": "dashscope"},
    {"id": "qwen",     "model": "qwen3.6-plus",           "label": "Qwen 3.6 Plus",  "provider": "dashscope"},
    {"id": "kimi",     "model": "kimi-k2.5",              "label": "Kimi K2.5",      "provider": "moonshot"},
    {"id": "grok",     "model": "grok-4.20",              "label": "Grok 4.20",      "provider": "xai"},
    {"id": "gpt4o",    "model": "gpt-4o",                 "label": "GPT-4o",         "provider": "openai"},
    {"id": "gpt54",    "model": "gpt-5.4",                "label": "GPT-5.4",        "provider": "openai"},
    {"id": "gpt54pro", "model": "gpt-5.4-pro",            "label": "GPT-5.4 Pro",    "provider": "openai_responses"},
    {"id": "gemini",   "model": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro", "provider": "google"},
]

JUDGE_SYSTEM = (
    "You are a senior staff engineer reviewing the design of an LLM-powered "
    "information extraction pipeline. Be critical but constructive. Identify "
    "real risks, not theoretical ones. If you recommend changes, be specific "
    "— show code deltas or prompt edits when possible. End with the required "
    "JSON verdict block exactly as specified; do not add trailing prose."
)

SCORE_KEYS = [
    "1_architecture",
    "2_triage_prompt",
    "3_extract_prompt",
    "4_resolver",
    "5_scorer",
    "6_apply_to_graph",
    "7_cost_efficiency",
]


# --------------------------------------------------------------------------
# Judge calls
# --------------------------------------------------------------------------
def _call_anthropic(model: str, user: str) -> str:
    r = client.messages.create(
        model=model,
        max_tokens=8000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in r.content if hasattr(b, "text"))
    if r.stop_reason == "max_tokens":
        text += "\n}"  # best-effort close
    return text


def _call_native(provider: str, model: str, user: str) -> str:
    if provider == "google":
        return _call_gemini(model, user)
    if provider == "openai_responses":
        return _call_openai_responses(model, user)
    cfg = NATIVE_APIS[provider]
    api_key = os.environ.get(cfg["key_env"], "")
    if not api_key:
        raise RuntimeError(f"Missing env var {cfg['key_env']}")
    tok_key = "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens"
    payload = json.dumps({
        "model": model,
        tok_key: 8000,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urlreq.Request(
        cfg["url"],
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlreq.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text.strip() and msg.get("reasoning_content"):
        text = msg["reasoning_content"]
    return text


def _call_openai_responses(model: str, user: str) -> str:
    api_key = os.environ.get("OPENAI_JUDGE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing env var OPENAI_JUDGE_API_KEY")
    payload = json.dumps({
        "model": model,
        "instructions": JUDGE_SYSTEM,
        "input": user,
        "max_output_tokens": 12000,
    }).encode("utf-8")
    req = urlreq.Request(
        "https://api.openai.com/v1/responses",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlreq.urlopen(req, timeout=600) as r:
        data = json.loads(r.read().decode("utf-8"))
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text", "")
    raise RuntimeError(f"No text in OpenAI Responses reply: {json.dumps(data)[:500]}")


def _call_gemini(model: str, user: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing env var GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = json.dumps({
        "systemInstruction": {"parts": [{"text": JUDGE_SYSTEM}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": 16000,
            "thinkingConfig": {"thinkingBudget": -1},
        },
    }).encode("utf-8")
    req = urlreq.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlreq.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode("utf-8"))
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini reply shape unexpected: {json.dumps(data)[:500]}")


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------
def _extract_verdict_json(text: str) -> dict | None:
    """Find the last top-level JSON block that looks like our verdict schema."""
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
        if not isinstance(obj, dict):
            continue
        if "verdict" in obj or "scores" in obj or "overall_score" in obj:
            return obj
    return None


def _call_judge(judge: dict, attempt: int = 1) -> tuple[str, dict | None, str | None]:
    """Returns (raw_text, verdict_json_or_None, error_or_None)."""
    try:
        prov = judge.get("provider", "anthropic")
        if prov == "anthropic":
            text = _call_anthropic(judge["model"], BUNDLE)
        else:
            text = _call_native(prov, judge["model"], BUNDLE)
        if not text or not text.strip():
            if attempt == 1:
                time.sleep(2)
                return _call_judge(judge, attempt=2)
            return "", None, "empty_response"
        verdict = _extract_verdict_json(text)
        return text, verdict, None
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        err = f"HTTP {e.code}: {body}"
        if attempt == 1:
            time.sleep(3)
            return _call_judge(judge, attempt=2)
        return "", None, err
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"
        if attempt == 1:
            time.sleep(3)
            return _call_judge(judge, attempt=2)
        return "", None, err


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
_DEDUP_MIN_LEN = 8


def _norm_item(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _merge_and_dedup(items: list[str]) -> list[tuple[str, int]]:
    """Dedup by substring — return [(canonical, count)] sorted by count desc.

    Items with normalized form that is a substring of another (>=8 chars overlap)
    are merged into the longer one.
    """
    cleaned = [(s.strip(), _norm_item(s)) for s in items if s and s.strip()]
    # Longest first so shorter substrings can collapse into longer ones
    cleaned.sort(key=lambda kv: -len(kv[1]))
    buckets: list[dict] = []  # {canonical, norm, members}
    for orig, norm in cleaned:
        if len(norm) < _DEDUP_MIN_LEN:
            # too short to reliably dedup; use it as its own bucket
            matched = False
            for b in buckets:
                if b["norm"] == norm:
                    b["members"].append(orig)
                    matched = True
                    break
            if not matched:
                buckets.append({"canonical": orig, "norm": norm, "members": [orig]})
            continue
        matched = False
        for b in buckets:
            if norm in b["norm"] or b["norm"] in norm:
                b["members"].append(orig)
                matched = True
                break
        if not matched:
            buckets.append({"canonical": orig, "norm": norm, "members": [orig]})
    out = [(b["canonical"], len(b["members"])) for b in buckets]
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def _pick_int(x, default=None):
    try:
        if isinstance(x, bool):
            return default
        return int(x)
    except Exception:
        return default


def _collect_score(verdicts: list[dict], key: str) -> tuple[list[int], float | None]:
    vals: list[int] = []
    for v in verdicts:
        scores = v.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        # accept a few variants ("1_architecture" vs "architecture")
        matched = None
        for k in scores.keys():
            if k == key or k.endswith(key[2:]) or key.endswith(k):
                matched = k
                break
        raw = scores.get(matched) if matched else None
        n = _pick_int(raw)
        if n is not None and 1 <= n <= 10:
            vals.append(n)
    avg = (sum(vals) / len(vals)) if vals else None
    return vals, avg


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = RESULTS_DIR / f"pulse-extraction-design-{ts}"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bench] design bundle: {BUNDLE_FILE} ({len(BUNDLE)} chars)")
    print(f"[bench] run dir: {run_dir}")
    print(f"[bench] judges: {len(JUDGES)}, parallel=5")
    print()

    results: dict[str, dict] = {}

    def _worker(judge: dict) -> tuple[str, dict]:
        t0 = time.time()
        text, verdict, err = _call_judge(judge)
        elapsed = time.time() - t0
        return judge["id"], {
            "judge": judge,
            "raw": text,
            "verdict": verdict,
            "error": err,
            "elapsed_s": round(elapsed, 1),
        }

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_worker, j): j for j in JUDGES}
        for fut in as_completed(futures):
            jid, res = fut.result()
            results[jid] = res
            label = res["judge"]["label"]
            if res["error"]:
                print(f"  [{label:16s}] FAIL ({res['elapsed_s']}s): {res['error']}")
            elif res["verdict"] is None:
                print(f"  [{label:16s}] parse_failed ({res['elapsed_s']}s, {len(res['raw'])} chars)")
            else:
                ov = res["verdict"].get("overall_score")
                vd = res["verdict"].get("verdict")
                print(f"  [{label:16s}] ok ({res['elapsed_s']}s): overall={ov} verdict={vd}")

    # --- Persist raw per-judge ---
    for jid, res in results.items():
        (raw_dir / f"{jid}.md").write_text(res["raw"] or "", encoding="utf-8")
        verdict_payload: dict = {
            "judge_id": jid,
            "label": res["judge"]["label"],
            "model": res["judge"]["model"],
            "provider": res["judge"]["provider"],
            "elapsed_s": res["elapsed_s"],
            "error": res["error"],
            "parse_failed": res["verdict"] is None and res["error"] is None,
            "verdict": res["verdict"],
        }
        (raw_dir / f"{jid}.json").write_text(
            json.dumps(verdict_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # --- Aggregate ---
    verdicts_ok = [r["verdict"] for r in results.values() if r["verdict"] is not None]
    score_rows: list[tuple[str, float | None, int, list[int]]] = []
    for key in SCORE_KEYS:
        vals, avg = _collect_score(verdicts_ok, key)
        score_rows.append((key, avg, len(vals), vals))

    overall_vals = []
    for v in verdicts_ok:
        n = _pick_int(v.get("overall_score"))
        if n is not None and 1 <= n <= 10:
            overall_vals.append(n)
    overall_avg = (sum(overall_vals) / len(overall_vals)) if overall_vals else None

    verdict_counts = {"ship": 0, "rework": 0, "rethink": 0, "other": 0}
    for v in verdicts_ok:
        vd = (v.get("verdict") or "").strip().lower()
        if vd in verdict_counts:
            verdict_counts[vd] += 1
        else:
            verdict_counts["other"] += 1

    all_bugs: list[str] = []
    all_improvs: list[str] = []
    for v in verdicts_ok:
        for item in (v.get("top_bugs") or []):
            if isinstance(item, str):
                all_bugs.append(item)
            elif isinstance(item, dict):
                t = item.get("bug") or item.get("text") or item.get("title") or json.dumps(item)
                if isinstance(t, str):
                    all_bugs.append(t)
        for item in (v.get("top_prompt_improvements") or v.get("top_improvements") or []):
            if isinstance(item, str):
                all_improvs.append(item)
            elif isinstance(item, dict):
                t = item.get("improvement") or item.get("text") or item.get("title") or json.dumps(item)
                if isinstance(t, str):
                    all_improvs.append(t)

    merged_bugs = _merge_and_dedup(all_bugs)
    merged_improvs = _merge_and_dedup(all_improvs)

    # --- summary.json ---
    summary = {
        "run_ts": ts,
        "bundle_path": str(BUNDLE_FILE),
        "bundle_chars": len(BUNDLE),
        "judges_total": len(JUDGES),
        "judges_ok": len(verdicts_ok),
        "judges_parse_failed": sum(
            1 for r in results.values() if r["verdict"] is None and r["error"] is None
        ),
        "judges_error": sum(1 for r in results.values() if r["error"]),
        "scores": {
            key: {"avg": avg, "n": n, "values": vals}
            for key, avg, n, vals in score_rows
        },
        "overall_score_avg": overall_avg,
        "overall_score_values": overall_vals,
        "verdict_distribution": verdict_counts,
        "top_bugs_merged": [{"text": t, "count": c} for t, c in merged_bugs],
        "top_prompt_improvements_merged": [{"text": t, "count": c} for t, c in merged_improvs],
        "per_judge": [
            {
                "id": jid,
                "label": r["judge"]["label"],
                "model": r["judge"]["model"],
                "elapsed_s": r["elapsed_s"],
                "error": r["error"],
                "parse_failed": r["verdict"] is None and r["error"] is None,
                "overall_score": (r["verdict"] or {}).get("overall_score") if r["verdict"] else None,
                "verdict": (r["verdict"] or {}).get("verdict") if r["verdict"] else None,
                "scores": (r["verdict"] or {}).get("scores") if r["verdict"] else None,
            }
            for jid, r in results.items()
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- report.md ---
    lines: list[str] = []
    lines.append("# Pulse Extraction Pipeline — Design Review Report")
    lines.append("")
    lines.append(f"Run: `{ts}`  •  Bundle: `{BUNDLE_FILE.name}` ({len(BUNDLE)} chars)  •  Judges: {len(JUDGES)}")
    lines.append("")
    lines.append(f"- OK judges: {len(verdicts_ok)}")
    lines.append(f"- Parse-failed: {summary['judges_parse_failed']}")
    lines.append(f"- Errored: {summary['judges_error']}")
    lines.append("")

    lines.append("## Average scores (1–10)")
    lines.append("")
    lines.append("| Category | Avg | n | Values |")
    lines.append("|---|---:|---:|---|")
    for key, avg, n, vals in score_rows:
        label = key.split("_", 1)[1].replace("_", " ")
        avg_s = f"{avg:.2f}" if avg is not None else "—"
        lines.append(f"| {key[0]}. {label} | {avg_s} | {n} | {vals} |")
    overall_s = f"{overall_avg:.2f}" if overall_avg is not None else "—"
    lines.append(f"| **overall** | **{overall_s}** | {len(overall_vals)} | {overall_vals} |")
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
    for jid, r in results.items():
        status = "ok" if r["verdict"] else ("parse_failed" if not r["error"] else "error")
        ov = (r["verdict"] or {}).get("overall_score") if r["verdict"] else None
        vd = (r["verdict"] or {}).get("verdict") if r["verdict"] else None
        ov_s = str(ov) if ov is not None else "—"
        vd_s = vd if vd else (r["error"] or "—")
        lines.append(
            f"| {r['judge']['label']} | `{r['judge']['model']}` | {ov_s} | {vd_s} | {r['elapsed_s']}s | {status} |"
        )
    lines.append("")

    lines.append(f"## Top bugs (merged across {len(verdicts_ok)} judges)")
    lines.append("")
    if merged_bugs:
        for text, count in merged_bugs[:15]:
            lines.append(f"- **[{count}×]** {text}")
    else:
        lines.append("_(no bugs extracted)_")
    lines.append("")

    lines.append(f"## Top prompt improvements (merged)")
    lines.append("")
    if merged_improvs:
        for text, count in merged_improvs[:15]:
            lines.append(f"- **[{count}×]** {text}")
    else:
        lines.append("_(no improvements extracted)_")
    lines.append("")

    # Key-judge full texts
    for key_id in ["gpt54pro", "opus"]:
        if key_id in results:
            r = results[key_id]
            lines.append(f"## Full response — {r['judge']['label']}")
            lines.append("")
            if r["error"]:
                lines.append(f"_Error: {r['error']}_")
            else:
                lines.append(r["raw"] or "_(empty)_")
            lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `raw/` — per-judge full markdown + verdict JSON")
    lines.append(f"- `summary.json` — machine-readable aggregate")
    lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # --- stdout summary ---
    print()
    print(f"[bench] wrote {run_dir}/report.md")
    print(f"[bench] wrote {run_dir}/summary.json")
    print(f"[bench] ok={len(verdicts_ok)}/{len(JUDGES)}  overall_avg={overall_s}  verdicts={verdict_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
