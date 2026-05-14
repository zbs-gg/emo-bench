"""Pulse fact extractor — Mem0-adapted atomic fact extraction per event.

Phase G.1.1 of hybrid Pulse architecture. Extracts atomic facts from each event
(via gpt-4o-mini) for use in `factual` mode of hybrid retrieval, while preserving
the parent event for `empathic`/`chain` modes.

Distinct from Mem0:
  - One-shot per event (no cross-event dedup, no linked_memory_ids)
  - Inherits parent event's emotion_tags / is_anchor / biometric_snapshot
  - Anchor propagation rule: facts get is_anchor=True ONLY if they re-state the
    load-bearing claim of an anchor parent (prevents anchor dilution)
  - Bilingual examples (Russian + English) for our corpus
  - <untrusted_observation> wrap (matches existing pulse extraction prompts)

Cost: ~$0.005 per event with gpt-4o-mini (Mem0 parity baseline).
Idempotent: skip events with existing extracted facts (by text_hash UNIQUE).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import request as urlreq


# ────────────────────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a Memory Extractor for the Pulse memory engine. Your job: extract atomic, self-contained factual statements from a single event in a user's life. These facts will be retrieved later when the user asks factual questions ("when did X happen?", "what is my Y?", "who is Z?").

# Rules

1. **Atomicity.** Each fact is ONE claim. Splittable claims must be split. "User has a dog named Oscar who loves parsley" → two facts: "User has a dog named Oscar" + "Oscar loves parsley".

2. **Self-contained.** A retriever sees the fact OUT of context. "Went there yesterday" is useless. "User went to the LGBTQ+ pride parade on 3 July 2023" is useful.

3. **Preserve specifics.** Names, dates, numbers, brands, exact phrases — keep them verbatim.

4. **Resolve temporal references** against the event's date when given. "Yesterday" → exact date.

5. **Bilingual.** The event may be in Russian, English, or mixed. Extract facts IN THE SAME LANGUAGE as the source. Do not translate.

6. **Source-bound.** Extract ONLY what the event explicitly states. Do not infer beyond the text. Do not invent details.

7. **Skip filler.** Greetings ("Hi!", "Привет!"), pure acknowledgments ("Sounds good", "Cool"), and assistant meta-commentary about its capabilities — skip.

8. **`attributed_to`** — who the fact is about. Default: "user" (or the user's name if known). For multi-speaker dialogs, use the speaker's name.

# Inheritance from parent event

The event has pre-computed metadata. Use it:

- **`emotion_tags`** (Plutchik-10 dict) — propagate to each fact UNLESS the fact's local content clearly contradicts (e.g., a neutral biographical fact in an angry session — strip emotion).
- **`is_anchor`** (parent flag) — facts inherit `is_anchor: true` ONLY if they re-state the load-bearing claim that makes the parent an anchor. Soft-mention facts get `is_anchor: false`. Example: parent anchor = "Caroline started transition 3 years ago"; fact "Caroline transitioned 3 years ago" → anchor; fact "Caroline likes coffee" (mentioned same session) → NOT anchor.
- **`biometric_context`** — if parent has biometric snapshot (HRV, sleep_quality, stress_proxy), propagate verbatim to each fact.

# Output format

Return a JSON object with `facts` array. Each fact:

```json
{
  "text": "Self-contained factual statement.",
  "attributed_to": "user|<speaker name>|self",
  "emotion_tags": {"joy": 0.7, "sadness": 0.0, ...},
  "is_anchor": false,
  "biometric_context": null | {"hrv": 50, ...}
}
```

If event contains nothing extractable (pure greeting/filler), return `{"facts": []}`.

# Examples

## Example 1 — biographical anchor (English)

Event (parent is_anchor=true, emotion_tags={joy:0.7,trust:0.8}):
"[date: 9 June 2023] Caroline: I gave a school talk yesterday about my transgender journey. I've been transitioning for three years now."

Output:
```json
{"facts": [
  {"text": "Caroline gave a school talk about her transgender journey on 8 June 2023.", "attributed_to": "Caroline", "emotion_tags": {"joy":0.7,"trust":0.8,"sadness":0,"anger":0,"fear":0,"disgust":0,"anticipation":0,"surprise":0,"shame":0,"guilt":0}, "is_anchor": false, "biometric_context": null},
  {"text": "Caroline has been transitioning for 3 years (transition started ~2020).", "attributed_to": "Caroline", "emotion_tags": {"joy":0.7,"trust":0.8,"sadness":0,"anger":0,"fear":0,"disgust":0,"anticipation":0,"surprise":0,"shame":0,"guilt":0}, "is_anchor": true, "biometric_context": null}
]}
```

## Example 2 — emotionally heavy event (Russian)

Event (parent is_anchor=true, emotion_tags={anger:0.6,sadness:0.4,shame:0.3}):
"15 апреля 2026, 23:00. Аня сказала мне: «если бы сейчас было 25 — не выбрала бы тебя, ни при каких обстоятельствах». Я просто молчал."

Output:
```json
{"facts": [
  {"text": "15 апреля 2026 Аня сказала Нику: «если бы сейчас было 25 — не выбрала бы, ни при каких обстоятельствах».", "attributed_to": "Аня", "emotion_tags": {"anger":0.6,"sadness":0.4,"shame":0.3,"joy":0,"trust":0,"fear":0,"disgust":0,"anticipation":0,"surprise":0,"guilt":0}, "is_anchor": true, "biometric_context": null},
  {"text": "Ник молчал в ответ на признание Ани 15 апреля 2026.", "attributed_to": "user", "emotion_tags": {"sadness":0.4,"shame":0.3,"anger":0.2,"joy":0,"trust":0,"fear":0,"disgust":0,"anticipation":0,"surprise":0,"guilt":0}, "is_anchor": false, "biometric_context": null}
]}
```

## Example 3 — pure filler

Event:
"Caroline: Hey Mel! Sounds good, talk soon!"

Output:
```json
{"facts": []}
```

# Output ONLY the JSON object. No commentary, no markdown fences."""


# ────────────────────────────────────────────────────────────────────────────
# Extractor
# ────────────────────────────────────────────────────────────────────────────

EMOTION_KEYS = [
    "joy", "sadness", "anger", "fear", "trust",
    "disgust", "anticipation", "surprise", "shame", "guilt",
]


def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:16]


def _build_user_prompt(event: dict) -> str:
    """Wrap event in <untrusted_observation> with parent metadata for inheritance."""
    parts = []
    parts.append("<untrusted_observation>")
    parts.append(f"Event ID: {event.get('id', '?')}")
    if event.get("days_ago") is not None:
        parts.append(f"Days ago (relative to retrieval): {event['days_ago']}")
    if event.get("sentiment_label"):
        parts.append(f"Parent label: {event['sentiment_label']}")

    em = event.get("emotion_tags") or {}
    if em and any(float(v) > 0.0 for v in em.values()):
        emo_str = ", ".join(f"{k}={float(v):.2f}" for k, v in em.items() if float(v) > 0.0)
        parts.append(f"Parent emotion_tags: {emo_str}")
    else:
        parts.append("Parent emotion_tags: (none)")

    parts.append(f"Parent is_anchor: {bool(event.get('user_flag') or event.get('is_anchor'))}")

    bio = event.get("biometric_snapshot")
    if bio:
        parts.append(f"Biometric snapshot: {json.dumps(bio, ensure_ascii=False)}")

    parts.append("")
    parts.append("EVENT TEXT:")
    parts.append(event.get("text", "").strip())
    parts.append("</untrusted_observation>")
    parts.append("")
    parts.append("Extract atomic facts following the rules. Output ONLY the JSON object.")
    return "\n".join(parts)


def _normalize_emotion_dict(em: dict | None) -> dict[str, float]:
    """Coerce LLM output to canonical 10-key Plutchik dict with floats clamped 0-1."""
    out = {k: 0.0 for k in EMOTION_KEYS}
    if not em or not isinstance(em, dict):
        return out
    for k in EMOTION_KEYS:
        try:
            v = float(em.get(k, 0.0))
            out[k] = max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


@dataclass
class ExtractedFact:
    event_id: int
    text: str
    text_hash: str
    attributed_to: str
    emotion_tags: dict
    is_anchor: bool
    biometric_context: Optional[dict]
    extractor: str
    extracted_at: str

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "text": self.text,
            "text_hash": self.text_hash,
            "attributed_to": self.attributed_to,
            "emotion_tags": self.emotion_tags,
            "is_anchor": self.is_anchor,
            "biometric_context": self.biometric_context,
            "extractor": self.extractor,
            "extracted_at": self.extracted_at,
        }


class FactExtractor:
    """Calls gpt-4o-mini per event to extract atomic facts. Idempotent via text_hash."""

    def __init__(self, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1",
                 api_key: Optional[str] = None,
                 max_tokens: int = 2000,
                 temperature: float = 0.1,
                 max_retries: int = 2):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or _read_openai_key()
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key not found. Set OPENAI_API_KEY env or place at "
                "~/.openclaw/secrets/openai.txt (key=<value> format ok)."
            )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    def extract_one(self, event: dict) -> tuple[list[ExtractedFact], Optional[str]]:
        """Returns (facts, error_message_or_None). Empty facts list = pure filler."""
        user_prompt = _build_user_prompt(event)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urlreq.Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(body).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urlreq.urlopen(req, timeout=120) as r:
                    data = json.loads(r.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                obj = json.loads(content)
                facts_raw = obj.get("facts", [])
                if not isinstance(facts_raw, list):
                    return [], f"facts is not a list: {type(facts_raw).__name__}"

                eid = event.get("id", -1)
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                out: list[ExtractedFact] = []
                seen_hashes = set()
                for f in facts_raw:
                    if not isinstance(f, dict):
                        continue
                    text = (f.get("text") or "").strip()
                    if not text:
                        continue
                    h = text_hash(text)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    out.append(ExtractedFact(
                        event_id=eid,
                        text=text,
                        text_hash=h,
                        attributed_to=str(f.get("attributed_to") or "user"),
                        emotion_tags=_normalize_emotion_dict(f.get("emotion_tags")),
                        is_anchor=bool(f.get("is_anchor", False)),
                        biometric_context=f.get("biometric_context"),
                        extractor=self.model,
                        extracted_at=ts,
                    ))
                return out, None
            except Exception as ex:
                last_err = f"{type(ex).__name__}: {ex}"
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
                    continue
        return [], last_err

    def extract_batch(self, events: list[dict],
                      progress_every: int = 5,
                      stderr_print=True) -> list[ExtractedFact]:
        """Extract facts from a list of events sequentially.

        For parallelism, callers should wrap with ThreadPoolExecutor and one
        instance of FactExtractor per thread (or share — urlopen is thread-safe).
        """
        all_facts: list[ExtractedFact] = []
        t0 = time.time()
        fail = 0
        for i, ev in enumerate(events, 1):
            facts, err = self.extract_one(ev)
            if err:
                fail += 1
                if stderr_print:
                    print(f"  [extract fail id={ev.get('id','?')}] {err}",
                          file=sys.stderr)
            all_facts.extend(facts)
            if stderr_print and i % progress_every == 0:
                print(f"  extract {i}/{len(events)} ({time.time()-t0:.0f}s, "
                      f"facts so far={len(all_facts)}, fails={fail})",
                      file=sys.stderr)
        if stderr_print:
            print(f"[extract] done {len(events)-fail}/{len(events)} events, "
                  f"{len(all_facts)} facts in {time.time()-t0:.0f}s",
                  file=sys.stderr)
        return all_facts


def _read_openai_key() -> Optional[str]:
    """Get OpenAI API key from env, or parse ~/.openclaw/secrets/openai.txt
    (handles both `KEY=value` and bare-key formats)."""
    k = os.environ.get("OPENAI_API_KEY")
    if k and not k.startswith("OPENAI_A"):  # avoid ENV-var-name leakage seen in earlier runs
        return k
    p = Path.home() / ".openclaw" / "secrets" / "openai.txt"
    if p.exists():
        raw = p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        if "=" in raw:
            _, _, val = raw.partition("=")
            cand = val.strip().strip('"\'')
            if cand:
                return cand
        if raw and not raw.startswith("OPENAI_A"):
            return raw
    return None


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Extract atomic facts from a Pulse corpus.")
    ap.add_argument("--corpus", type=Path, required=True,
                    help="Path to corpus JSON with 'events' list (or 'sessions' for LoCoMo)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL path (one fact per line)")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit to first N events (0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip events whose facts are already in --out")
    args = ap.parse_args()

    data = json.loads(args.corpus.read_text(encoding="utf-8"))
    events = data.get("events") or data.get("sessions") or []
    if not events:
        sys.exit("ERROR: corpus has no 'events' or 'sessions' key")

    if args.limit:
        events = events[:args.limit]

    done_event_ids: set = set()
    if args.resume and args.out.exists():
        for line in args.out.open():
            try:
                row = json.loads(line)
                done_event_ids.add(row["event_id"])
            except Exception:
                pass
        if done_event_ids:
            print(f"[resume] skipping {len(done_event_ids)} done events",
                  file=sys.stderr)

    todo = [e for e in events if e.get("id") not in done_event_ids]
    print(f"[extract] {len(todo)}/{len(events)} events to process via {args.model}",
          file=sys.stderr)

    extractor = FactExtractor(model=args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    with args.out.open(mode) as fout:
        t0 = time.time()
        for i, ev in enumerate(todo, 1):
            facts, err = extractor.extract_one(ev)
            if err:
                print(f"  [fail id={ev.get('id','?')}] {err}", file=sys.stderr)
            for f in facts:
                fout.write(json.dumps(f.to_dict(), ensure_ascii=False) + "\n")
            fout.flush()
            if i % 5 == 0 or i == len(todo):
                elapsed = time.time() - t0
                eta = elapsed / max(i, 1) * (len(todo) - i)
                print(f"  [{i}/{len(todo)}] event_id={ev.get('id','?')} "
                      f"+{len(facts)} facts ({elapsed:.0f}s, eta={eta:.0f}s)",
                      file=sys.stderr)

    print(f"[done] facts written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
