"""Pulse hybrid (Phase G) on LoCoMo benchmark.

Adapts run_pulse_locomo.py to use RetrievalV3.retrieve_hybrid() with router +
per-sample fact subset. Each LoCoMo conversation → its sessions become events;
its facts (extracted earlier via extract_locomo_facts.py) are indexed; per-QA
the router classifies → factual mode searches facts → parent session_id, or
empathic mode does cosine on session texts.

Reuses kimi_chat (Qwen3-Max) for QA, same evaluator as run_pulse_locomo.py
(token F1).

Usage:
  python run_pulse_locomo_hybrid.py \\
    --data <locomo10.json> \\
    --facts /tmp/pulse_facts_locomo.jsonl \\
    --out <hyps.jsonl> --provider qwen
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


def parse_locomo_date(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ["%I:%M %p on %d %B, %Y", "%I:%M%p on %d %B, %Y"]:
        try:
            return datetime.strptime(s, fmt)
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


def build_events(sample: dict) -> tuple[list[dict], dict[int, str]]:
    """Sessions → events list with synthetic ids 0..N-1.
    Returns (events, idx_to_session_id_map)."""
    sessions = get_sessions(sample)
    parsed = [parse_locomo_date(d) for _, _, d in sessions]
    valid = [d for d in parsed if d is not None]
    max_d = max(valid) if valid else None

    events = []
    idx_to_sid = {}
    for i, (sid, text, _date_str) in enumerate(sessions):
        d = parsed[i]
        days_ago = max(0.0, (max_d - d).total_seconds() / 86400.0) if (d and max_d) else float(len(sessions)-1-i)
        events.append({
            "id": i,
            "text": text,
            "days_ago": days_ago,
            "emotion_tags": {},
            "user_flag": False,
        })
        idx_to_sid[i] = sid
    return events, idx_to_sid


def filter_facts_for_sample(all_facts: list[dict], sample_id: str,
                            sid_to_event_idx: dict[str, int]) -> list[dict]:
    """Subset facts to this sample, remap session_id → event_id (synthetic)."""
    out = []
    for f in all_facts:
        if f.get("sample_id") != sample_id:
            continue
        sid = f.get("session_id")
        if sid not in sid_to_event_idx:
            continue
        out.append({**f, "event_id": sid_to_event_idx[sid]})
    return out


async def answer_one(client, question, retrieved, sem, model):
    async with sem:
        history = "\n\n".join(f"=== {sid} ({sdate}) ===\n{stext}"
                              for sid, stext, sdate in retrieved)
        prompt = f"Retrieved conversation sessions:\n\n{history}\n\nQuestion: {question}\n\nAnswer:"
        return await kimi_chat(client, model, QA_SYSTEM, prompt, max_tokens=3000)


async def process_sample(client, sample, all_facts, top_k, qa_model, sem):
    events, idx_to_sid = build_events(sample)
    if not events:
        return []

    sid_to_idx = {sid: i for i, sid in idx_to_sid.items()}
    sample_facts = filter_facts_for_sample(all_facts, sample["sample_id"], sid_to_idx)

    engine = RetrievalV3(events, use_llm_query_emo=False, query_emo_provider="qwen")
    if sample_facts:
        engine.index_facts(sample_facts)

    sessions = get_sessions(sample)
    text_by_idx = {i: events[i]["text"] for i in range(len(events))}
    date_by_idx = {i: sessions[i][2] for i in range(len(sessions))}

    qas = sample.get("qa", [])
    tasks = []
    mode_counts = {"factual": 0, "empathic": 0, "chain": 0}
    for qa in qas:
        question = qa["question"]
        ids, decision = engine.retrieve_hybrid(question, user_state=None, top_k=top_k,
                                               mode="auto", return_decision=True)
        used_mode = decision.mode if decision else "empathic"
        mode_counts[used_mode] = mode_counts.get(used_mode, 0) + 1
        retrieved = [(idx_to_sid[i], text_by_idx[i], date_by_idx[i]) for i in ids]
        rids = [idx_to_sid[i] for i in ids]

        async def pack(qa=qa, retrieved=retrieved, rids=rids, used_mode=used_mode):
            hyp = await answer_one(client, qa["question"], retrieved, sem, qa_model)
            return qa, rids, hyp, used_mode

        tasks.append(pack())

    packed = await asyncio.gather(*tasks)
    results = []
    for qa, rids, hyp, used_mode in packed:
        results.append({
            "sample_id": sample["sample_id"],
            "question": qa["question"],
            "answer": qa.get("answer", ""),
            "adversarial_answer": qa.get("adversarial_answer", ""),
            "evidence": qa.get("evidence", []),
            "category": qa.get("category"),
            "hypothesis": hyp,
            "retrieved_session_ids": rids,
            "mode_used": used_mode,
        })
    print(f"  [{sample['sample_id']}] modes: {mode_counts}", file=sys.stderr)
    return results


async def run(data_file: Path, facts_file: Path, out_file: Path,
              top_k: int, qa_model: str, provider: str, concurrency: int, limit: int):
    client = llm_client(provider)
    data = json.load(open(data_file))
    if limit:
        data = data[:limit]

    all_facts = []
    if facts_file and facts_file.exists():
        for line in facts_file.open():
            if line.strip():
                all_facts.append(json.loads(line))
    print(f"[facts] {len(all_facts)} facts loaded from {facts_file}",
          file=sys.stderr)

    done = set()
    if out_file.exists():
        for line in out_file.open():
            if line.strip():
                row = json.loads(line)
                done.add((row["sample_id"], row["question"]))
    if done:
        print(f"[resume] skipping {len(done)} done QAs", file=sys.stderr)

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    print(f"LoCoMo Pulse-hybrid: {len(data)} samples, qa={qa_model}",
          file=sys.stderr)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("a") as f:
        for i, sample in enumerate(data, 1):
            sid = sample["sample_id"]
            qas = sample.get("qa", [])
            qas_to_do = [qa for qa in qas if (sid, qa["question"]) not in done]
            if not qas_to_do:
                print(f"[{i}/{len(data)}] {sid} all cached, skip",
                      file=sys.stderr)
                continue
            print(f"[{i}/{len(data)}] {sid}: {len(qas_to_do)} QAs",
                  file=sys.stderr)
            sample_filtered = dict(sample)
            sample_filtered["qa"] = qas_to_do
            try:
                results = await process_sample(client, sample_filtered, all_facts,
                                                top_k, qa_model, sem)
            except Exception as ex:
                print(f"  ERR {sid}: {str(ex)[:200]}", file=sys.stderr)
                continue
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            eta = el / max(i, 1) * (len(data) - i)
            print(f"  +{len(results)} QAs (elapsed={el:.0f}s, eta={eta:.0f}s)",
                  file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--provider", default="qwen", choices=list(PROVIDERS.keys()))
    p.add_argument("--qa-model", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    qa = args.qa_model or PROVIDERS[args.provider]["default_model"]
    asyncio.run(run(args.data, args.facts, args.out, args.top_k,
                    qa, args.provider, args.concurrency, args.limit))


if __name__ == "__main__":
    main()
