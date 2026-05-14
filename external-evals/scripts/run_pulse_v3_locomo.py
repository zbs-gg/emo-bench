"""Pulse v3 (FULL stack: emotion + anchor + date) on LoCoMo.

Difference from run_pulse_locomo.py:
- Sessions are turned into events with emotion_tags / user_flag / days_ago
  populated from a verdicts file (see prepare_locomo_pulse_input.py +
  in-chat Opus tagging)
- Retrieval uses RetrievalV3 (cosine × decay × emotion × state × anchor × date)
- QA reader stays the same (Qwen3-Max)

Without verdicts file: dies with error — refuses to silently fall back to v2_pure
(which is what bench has been measuring for the past month). Honest bench.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import llm_client, kimi_chat, PROVIDERS
from retrieval_v3 import RetrievalV3, UserState


QA_SYSTEM = (
    "You answer questions about a past conversation between two people. "
    "Use ONLY the provided retrieved sessions as context. Be concise (1-2 sentences). "
    "If the information isn't in the retrieved sessions, say 'I don't know'. "
    "For date questions, give the exact date. For names, give the exact name. "
    "Do not hallucinate."
)


def parse_locomo_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    fmts = ["%I:%M %p on %d %B, %Y", "%I:%M%p on %d %B, %Y"]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return None


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
        text = "\n".join(parts)[:8000]
        out.append((k, text, date))
    return out


def build_events_for_sample(sample: dict, verdicts: dict) -> list[dict]:
    """Build RetrievalV3 events from a LoCoMo sample using verdicts.

    days_ago = days from latest session to each session (latest=0).
    """
    sample_id = sample["sample_id"]
    sessions = get_sessions(sample)

    parsed_dates = [parse_locomo_date(d) for _, _, d in sessions]
    valid_dates = [d for d in parsed_dates if d is not None]
    if not valid_dates:
        # No dates: use index order
        max_date = None
    else:
        max_date = max(valid_dates)

    events = []
    for i, (sid, text, date_str) in enumerate(sessions):
        verdict_key = f"{sample_id}|{sid}"
        v = verdicts.get(verdict_key)
        if not v:
            print(f"  WARN: missing verdict for {verdict_key}", file=sys.stderr)
            continue
        d = parsed_dates[i]
        if d and max_date:
            days_ago = max(0.0, (max_date - d).total_seconds() / 86400.0)
        else:
            days_ago = float(len(sessions) - 1 - i)
        events.append({
            "id": i,
            "text": text,
            "days_ago": days_ago,
            "emotion_tags": v.get("emotion_tags", {}),
            "user_flag": bool(v.get("is_anchor", False)),
            "sentiment_label": v.get("sentiment_label", ""),
            "predecessor_ids": [],
            "_session_id": sid,
        })
    return events


async def answer_one(client, question, retrieved_sessions, sem, model):
    async with sem:
        history = "\n\n".join(
            f"=== {sid} ===\n{stext}" for sid, stext in retrieved_sessions
        )
        prompt = f"Retrieved conversation sessions:\n\n{history}\n\nQuestion: {question}\n\nAnswer:"
        return await kimi_chat(client, model, QA_SYSTEM, prompt, max_tokens=3000)


async def process_sample(client, sample, verdicts, top_k, qa_model, sem, query_emo_provider):
    events = build_events_for_sample(sample, verdicts)
    if not events:
        return []

    engine = RetrievalV3(
        events,
        query_emo_provider=query_emo_provider,
        use_llm_query_emo=True,
    )
    sid_by_event_id = {e["id"]: e["_session_id"] for e in events}
    text_by_event_id = {e["id"]: e["text"] for e in events}

    qas = sample.get("qa", [])
    tasks = []
    for qa in qas:
        question = qa["question"]
        ids = engine.retrieve(question, user_state=UserState(), top_k=top_k)
        retrieved = [(sid_by_event_id[i], text_by_event_id[i]) for i in ids]
        rids = [sid_by_event_id[i] for i in ids]

        async def pack(qa=qa, retrieved=retrieved, rids=rids):
            hyp = await answer_one(client, qa["question"], retrieved, sem, qa_model)
            return qa, rids, hyp

        tasks.append(pack())

    packed = await asyncio.gather(*tasks)
    results = []
    for qa, rids, hyp in packed:
        results.append({
            "sample_id": sample["sample_id"],
            "question": qa["question"],
            "answer": qa.get("answer", ""),
            "adversarial_answer": qa.get("adversarial_answer", ""),
            "evidence": qa.get("evidence", []),
            "category": qa.get("category"),
            "hypothesis": hyp,
            "retrieved_session_ids": rids,
        })
    return results


async def run(data_file: Path, verdicts_file: Path, out_file: Path,
              top_k: int, qa_model: str, provider: str, concurrency: int, limit: int):
    client = llm_client(provider)
    data = json.load(open(data_file))
    verdicts = json.load(open(verdicts_file))
    print(f"[verdicts] {len(verdicts)} session-level annotations loaded", file=sys.stderr)
    if limit:
        data = data[:limit]

    done = set()
    if out_file.exists():
        for line in out_file.open():
            if line.strip():
                row = json.loads(line)
                done.add((row["sample_id"], row["question"]))

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    print(f"LoCoMo Pulse-v3-FULL: {len(data)} samples, cached={len(done)}", file=sys.stderr)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("a") as f:
        for i, sample in enumerate(data, 1):
            sid = sample["sample_id"]
            qas = sample.get("qa", [])
            qas_to_do = [qa for qa in qas
                         if (sid, qa["question"]) not in done]
            if not qas_to_do:
                print(f"[{i}/{len(data)}] {sid} all cached, skip", file=sys.stderr)
                continue
            print(f"[{i}/{len(data)}] {sid}: {len(qas_to_do)} QAs", file=sys.stderr)
            sample_filtered = dict(sample)
            sample_filtered["qa"] = qas_to_do
            try:
                results = await process_sample(client, sample_filtered, verdicts,
                                                top_k, qa_model, sem, provider)
            except Exception as ex:
                print(f"  ERR {sid}: {str(ex)[:200]}", file=sys.stderr)
                continue
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"  +{len(results)} QAs. elapsed={el:.0f}s "
                  f"eta={(len(data)-i)/max(i,1)*el:.0f}s", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--verdicts", type=Path, required=True,
                   help="JSON {sample_id|sid: {emotion_tags, is_anchor, sentiment_label}}")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--provider", default="qwen", choices=list(PROVIDERS.keys()))
    p.add_argument("--qa-model", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    qa = args.qa_model or PROVIDERS[args.provider]["default_model"]
    asyncio.run(run(args.data, args.verdicts, args.out, args.top_k,
                    qa, args.provider, args.concurrency, args.limit))


if __name__ == "__main__":
    main()
