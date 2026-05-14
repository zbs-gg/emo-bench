"""Emotion classifier: tag each event's emotion_tags (Plutchik-10) via LLM.

Reference document: datasets/emotion-taxonomy.md (loaded into system prompt).

Strategy:
  - SKIP events that already have non-zero emotion_tags (manually-set disambig events + anchors)
  - For remaining events: Qwen Max call per event with taxonomy in system prompt
  - Rate-limited (2 parallel) to stay under DashScope QPM
  - Idempotent: writes in-place, re-runnable
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request as urlreq

sys.path.insert(0, str(Path(__file__).parent))
from common import secret, PROVIDERS

EMOTION_KEYS = [
    "joy", "sadness", "anger", "fear", "trust",
    "disgust", "anticipation", "surprise", "shame", "guilt",
]


def _load_taxonomy() -> str:
    """Load emotion-taxonomy.md as reference for the classifier."""
    tax_path = Path(__file__).parent.parent.parent / "datasets" / "emotion-taxonomy.md"
    if tax_path.exists():
        return tax_path.read_text(encoding="utf-8")
    return ""


SYSTEM_PROMPT = """You tag the emotional texture of life events for an empathic memory bench.
Use Plutchik-8 + shame + guilt (10 dimensions). For each emotion rate intensity 0.0–1.0:
  0.0 = absent
  0.1–0.3 = trace
  0.4–0.6 = present
  0.7–0.9 = dominant
  1.0 = overwhelming, defining

IMPORTANT rules:
  - Tag what is IN the event for the protagonist, NOT what a reader might feel about it
  - Multi-emotion is NORMAL. Events that end up mono-emotion are usually mundane facts
  - Shame = identity wound ("I am bad"). Guilt = action-regret ("I did bad, can repair")
  - Don't under-tag shame and guilt — they're load-bearing in empathic retrieval

Output ONLY a JSON object with the 10 keys, no markdown, no explanation:
{"joy":0.0,"sadness":0.0,"anger":0.0,"fear":0.0,"trust":0.0,"disgust":0.0,"anticipation":0.0,"surprise":0.0,"shame":0.0,"guilt":0.0}
"""


def _has_real_tags(event: dict) -> bool:
    """True if event already has non-zero emotion_tags (don't overwrite)."""
    tags = event.get("emotion_tags") or {}
    return any(float(v) > 0.0 for v in tags.values())


def classify_one(event: dict, provider: str, model: str, taxonomy: str,
                 max_retries: int = 2) -> tuple[int, dict[str, float] | str]:
    """Call LLM to classify one event. Returns (event_id, tags_dict_or_error_str)."""
    cfg = PROVIDERS[provider]
    api_key = secret(cfg["key_file"])

    # Include taxonomy for anchor context
    user = (
        f"Event text:\n{event['text']}\n\n"
        f"Label: {event.get('sentiment_label','?')}\n"
        f"Category: {event.get('category','legacy')}\n"
        f"Sentiment (-3..+3): {event.get('sentiment', 0)}\n\n"
        "Output the 10-key JSON:"
    )

    system = SYSTEM_PROMPT
    if taxonomy:
        system = system + "\n\n--- REFERENCE DOCUMENT (emotion-taxonomy.md) ---\n" + taxonomy

    for attempt in range(max_retries + 1):
        payload = json.dumps({
            "model": model,
            "max_tokens": 1000,
            "temperature": 1.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        req = urlreq.Request(
            f"{cfg['base_url']}/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlreq.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode())
            text = data["choices"][0]["message"].get("content") or ""
            m = re.search(r"\{[^}]*\}", text, re.DOTALL)
            if not m:
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return (event["id"], f"no_json: {text[:200]}")
            obj = json.loads(m.group())
            tags = {k: round(float(obj.get(k, 0.0)), 2) for k in EMOTION_KEYS}
            return (event["id"], tags)
        except Exception as ex:
            if attempt < max_retries:
                time.sleep(5 * (attempt + 1))
                continue
            return (event["id"], f"error: {str(ex)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--provider", type=str, default="qwen",
                    help=f"One of {list(PROVIDERS)}")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--only-missing", action="store_true",
                    help="Skip events that already have non-zero emotion_tags (default: false — re-tag all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Print plan, don't call LLM")
    args = ap.parse_args()

    data = json.loads(args.corpus.read_text(encoding="utf-8"))
    events = data["events"]
    model = args.model or PROVIDERS[args.provider]["default_model"]
    taxonomy = _load_taxonomy()

    to_tag = []
    for ev in events:
        if args.only_missing and _has_real_tags(ev):
            continue
        to_tag.append(ev)
        if args.limit and len(to_tag) >= args.limit:
            break

    print(f"Corpus: {args.corpus.name}", file=sys.stderr)
    print(f"Events total: {len(events)}; to tag: {len(to_tag)} "
          f"({'--only-missing' if args.only_missing else 'all'})", file=sys.stderr)
    print(f"Provider: {args.provider}, model: {model}", file=sys.stderr)

    if args.dry_run:
        print(f"[dry-run] would tag events: {[e['id'] for e in to_tag]}", file=sys.stderr)
        return

    # Build id → event map for in-place update
    eid_map = {e["id"]: e for e in events}
    t0 = time.time()
    done = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(classify_one, ev, args.provider, model, taxonomy): ev["id"]
                for ev in to_tag}
        for f in as_completed(futs):
            eid = futs[f]
            try:
                _, result = f.result()
            except Exception as ex:
                result = f"executor_error: {ex}"
            if isinstance(result, dict):
                eid_map[eid]["emotion_tags"] = result
                done += 1
                top = sorted(result.items(), key=lambda kv: -kv[1])[:3]
                print(f"  [{done}/{len(to_tag)}] id={eid} top={top} "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr)
            else:
                failed.append((eid, result))
                print(f"  FAIL id={eid}: {result}", file=sys.stderr)

    # Save
    args.corpus.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {args.corpus}: {done}/{len(to_tag)} tagged, {len(failed)} failed",
          file=sys.stderr)
    if failed:
        print("Failed IDs:", [eid for eid, _ in failed], file=sys.stderr)


if __name__ == "__main__":
    main()
