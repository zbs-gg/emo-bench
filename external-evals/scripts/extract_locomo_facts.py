"""Extract facts from LoCoMo sessions for hybrid Pulse retrieval.

Wraps fact_extractor to handle LoCoMo's session structure:
  conversation/{session_N: [turns], session_N_date_time: ...}

Each session becomes one "event" with synthetic id, text=concat(turns), date.
Output: JSONL of facts per session_id (event_id = "{sample_id}|{session_id}").
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fact_extractor import FactExtractor


def session_to_event(sample_id: str, session_id: str, turns: list[dict],
                     date: str, event_idx: int) -> dict:
    parts = [f"[date: {date}]"] if date else []
    for turn in turns:
        speaker = turn.get("speaker", "?")
        text = turn.get("text", "")
        parts.append(f"{speaker}: {text}")
    return {
        "id": event_idx,
        "_sample_id": sample_id,
        "_session_id": session_id,
        "text": "\n".join(parts)[:6000],
        "days_ago": None,
        "emotion_tags": {},
        "user_flag": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="LoCoMo locomo10.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL of facts")
    ap.add_argument("--samples", type=str, default="",
                    help="Comma-separated sample_ids (default: all 10)")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help="Limit total sessions (smoke test)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data = json.load(open(args.data))
    sample_filter = set(s.strip() for s in args.samples.split(",") if s.strip()) if args.samples else None

    events = []
    event_idx = 0
    for sample in data:
        sid = sample["sample_id"]
        if sample_filter and sid not in sample_filter:
            continue
        conv = sample["conversation"]
        keys = sorted([k for k in conv.keys()
                       if k.startswith("session_") and not k.endswith("_date_time")],
                      key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0)
        for k in keys:
            turns = conv[k]
            if not isinstance(turns, list):
                continue
            date = conv.get(f"{k}_date_time", "")
            events.append(session_to_event(sid, k, turns, date, event_idx))
            event_idx += 1
            if args.limit_sessions and event_idx >= args.limit_sessions:
                break
        if args.limit_sessions and event_idx >= args.limit_sessions:
            break

    print(f"[locomo] {len(events)} sessions to extract facts from", file=sys.stderr)

    done_event_ids = set()
    if args.resume and args.out.exists():
        for line in args.out.open():
            try:
                row = json.loads(line)
                done_event_ids.add(row["event_id"])
            except Exception:
                pass
        if done_event_ids:
            print(f"[resume] skipping {len(done_event_ids)} done sessions",
                  file=sys.stderr)

    todo = [e for e in events if e["id"] not in done_event_ids]
    print(f"[extract] {len(todo)}/{len(events)} sessions to process via {args.model}",
          file=sys.stderr)

    extractor = FactExtractor(model=args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    import time
    t0 = time.time()
    with args.out.open(mode) as fout:
        for i, ev in enumerate(todo, 1):
            facts, err = extractor.extract_one(ev)
            if err:
                print(f"  [fail id={ev['id']} {ev['_sample_id']}|{ev['_session_id']}] {err}",
                      file=sys.stderr)
            for f in facts:
                row = f.to_dict()
                # Tag with sample/session for downstream LoCoMo run
                row["sample_id"] = ev["_sample_id"]
                row["session_id"] = ev["_session_id"]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 5 == 0 or i == len(todo):
                el = time.time() - t0
                eta = el / max(i, 1) * (len(todo) - i)
                print(f"  [{i}/{len(todo)}] {ev['_sample_id']}|{ev['_session_id']} "
                      f"+{len(facts)} facts ({el:.0f}s, eta={eta:.0f}s)",
                      file=sys.stderr)

    print(f"[done] facts written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
