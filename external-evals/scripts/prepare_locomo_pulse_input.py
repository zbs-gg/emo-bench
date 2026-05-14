"""Prepare LoCoMo sessions as a batched input file for in-chat Opus
emotion classification + anchor selection (no API spend — Opus runs in
the chat session itself).

Output: /tmp/locomo_pulse_input.jsonl — one batch per line:
  {"batch_id": int, "sample_id": str, "sessions": [
      {"sid": "session_1", "date": "2024-...", "text": "..."}, ...]}

In-chat Opus reads each batch, returns a verdict file:
  {"sample_id|sid": {"emotion_tags": {...10}, "is_anchor": bool,
                     "sentiment_label": str}}

Then merge_locomo_pulse_into_corpus.py produces enriched-locomo.json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def get_sessions(sample: dict) -> list[tuple[str, str, str]]:
    conv = sample["conversation"]
    out = []
    keys = sorted([k for k in conv.keys()
                   if k.startswith("session_") and not k.endswith("_date_time")],
                  key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0)
    for k in keys:
        turns = conv[k]
        if not isinstance(turns, list):
            continue
        date_key = f"{k}_date_time"
        date = conv.get(date_key, "")
        parts = [f"[date: {date}]"] if date else []
        for turn in turns:
            speaker = turn.get("speaker", "?")
            text = turn.get("text", "")
            parts.append(f"{speaker}: {text}")
        text = "\n".join(parts)[:6000]  # cap per-session text
        out.append((k, text, date))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="LoCoMo locomo10.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL (one batch per line)")
    ap.add_argument("--samples", type=str, default="",
                    help="Comma-separated sample_ids (default: all 10)")
    ap.add_argument("--batch-size", type=int, default=7,
                    help="Sessions per batch")
    args = ap.parse_args()

    data = json.load(open(args.data))
    sample_filter = set(s.strip() for s in args.samples.split(",") if s.strip()) if args.samples else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        bid = 0
        total_sessions = 0
        for sample in data:
            sid = sample["sample_id"]
            if sample_filter and sid not in sample_filter:
                continue
            sessions = get_sessions(sample)
            for i in range(0, len(sessions), args.batch_size):
                chunk = sessions[i:i + args.batch_size]
                batch = {
                    "batch_id": bid,
                    "sample_id": sid,
                    "sessions": [
                        {"sid": s[0], "date": s[2], "text": s[1]}
                        for s in chunk
                    ],
                }
                f.write(json.dumps(batch, ensure_ascii=False) + "\n")
                bid += 1
                total_sessions += len(chunk)
            print(f"{sid}: {len(sessions)} sessions → "
                  f"{(len(sessions) + args.batch_size - 1) // args.batch_size} batches")
        print(f"---\n{bid} batches, {total_sessions} sessions → {args.out}")


if __name__ == "__main__":
    main()
