"""Mem0 on LongMemEval_S — apples-to-apples vs Pulse v2_pure run_pulse_lme.

Per-question: Mem0 ingests haystack_sessions for that question, then searches +
QA via Qwen. Each question has its own user_id (question_id) so memory is isolated
per question.

Usage:
  OPENAI_API_KEY=... python run_mem0_lme.py --data <lme_s.json> --out <hyps.jsonl> \\
      --provider qwen
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import llm_client, kimi_chat, PROVIDERS


QA_SYSTEM = ("You answer using only provided memories. "
             "Be concise (1-2 sentences). If info missing, say so. "
             "For temporal questions, use timestamps. Do not hallucinate.")


def concat_session(session, max_chars=8000):
    return "\n".join(f"[{t.get('role','?')}] {t.get('content','')}" for t in session)[:max_chars]


async def answer_one(client, question, qdate, memories, sem, model):
    async with sem:
        history = "\n\n".join(f"=== Memory {i+1} ===\n{m}" for i, m in enumerate(memories))
        prompt = (f"[Question asked on {qdate}]\n\nMemories:\n\n{history}\n\n"
                  f"Question: {question}\n\nAnswer:")
        return await kimi_chat(client, model, QA_SYSTEM, prompt, max_tokens=3000)


async def process_one(memory, client, entry, top_k, qa_model, sem):
    qid = entry["question_id"]
    sessions = entry["haystack_sessions"]
    dates = entry.get("haystack_dates", [""] * len(sessions))
    session_ids = entry.get("haystack_session_ids", list(range(len(sessions))))

    # Wipe per-question memory
    try:
        memory.delete_all(user_id=qid)
    except Exception:
        pass

    # Ingest haystack
    for sid, sess, date in zip(session_ids, sessions, dates):
        text = concat_session(sess)
        try:
            memory.add(text, user_id=qid, metadata={"session_id": sid, "date": date})
        except Exception as ex:
            print(f"  [mem0 add error qid={qid} sid={sid}] {ex}", file=sys.stderr)

    # Search + QA
    try:
        r = memory.search(entry["question"], filters={"user_id": qid}, limit=top_k)
        mems = [item.get("memory", "") for item in r.get("results", [])[:top_k]]
    except Exception as ex:
        print(f"  [mem0 search error qid={qid}] {ex}", file=sys.stderr)
        mems = []

    hyp = await answer_one(client, entry["question"], entry.get("question_date", ""),
                            mems, sem, qa_model)
    return {
        "question_id": qid,
        "hypothesis": hyp,
        "retrieved_memories": mems[:3],
        "evidence_session_ids": entry.get("answer_session_ids", []),
    }


async def run(data_file: Path, out_file: Path, top_k: int, qa_model: str,
              provider: str, concurrency: int, limit: int):
    from mem0 import Memory
    # Separate storage path so this can run concurrently with LoCoMo (default /tmp/qdrant)
    memory = Memory.from_config({
        "vector_store": {
            "provider": "qdrant",
            "config": {"path": "/tmp/qdrant_lme", "collection_name": "lme_mem0"}
        }
    })
    print(f"[mem0] initialized flat (storage=/tmp/qdrant_lme)", file=sys.stderr)

    client = llm_client(provider)
    data = json.load(open(data_file))
    if limit:
        data = data[:limit]
    print(f"[lme] {len(data)} questions", file=sys.stderr)

    sem = asyncio.Semaphore(concurrency)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done = set()
    if out_file.exists():
        for line in out_file.open():
            try:
                done.add(json.loads(line)["question_id"])
            except Exception:
                pass
    if done:
        print(f"[resume] skipping {len(done)} done", file=sys.stderr)

    with out_file.open("a") as fout:
        for i, entry in enumerate(data, 1):
            if entry["question_id"] in done:
                continue
            print(f"[q {i}/{len(data)}] {entry['question_id']}", file=sys.stderr)
            try:
                r = await process_one(memory, client, entry, top_k, qa_model, sem)
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as ex:
                print(f"  [process error] {ex}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--provider", type=str, default="qwen")
    ap.add_argument("--qa-model", type=str, default="")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: set OPENAI_API_KEY (Mem0 fact extraction uses gpt-4o-mini)")

    qa_model = args.qa_model or PROVIDERS[args.provider]["default_model"]
    asyncio.run(run(args.data, args.out, args.top_k, qa_model, args.provider,
                    args.concurrency, args.limit))


if __name__ == "__main__":
    main()
