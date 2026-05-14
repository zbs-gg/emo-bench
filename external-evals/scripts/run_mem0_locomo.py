"""Mem0 on LoCoMo benchmark — apples-to-apples with our Pulse v2_pure run.

Uses:
- Mem0 (mem0ai library) for memory storage and retrieval
- Same Qwen3-Max as QA reader (matches our Pulse run)
- Same evaluator (token F1) for direct comparison

Cost: ~$5-10 (Mem0 fact extraction via gpt-4o-mini default + Qwen QA).

Usage:
  OPENAI_API_KEY=... python run_mem0_locomo.py \\
      --data <locomo10.json> --out <hyps.jsonl> --provider qwen [--graph]
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


QA_SYSTEM = (
    "You answer questions about a past conversation between two people. "
    "Use ONLY the provided memories as context. Be concise (1-2 sentences). "
    "If the information isn't in the memories, say 'I don't know'. "
    "For date questions, give the exact date. For names, give the exact name. "
    "Do not hallucinate."
)


def build_session_text(turns: list[dict], session_date: str) -> str:
    """Concat session turns into one text block — same as Pulse run."""
    parts = [f"[session date: {session_date}]"] if session_date else []
    for turn in turns:
        speaker = turn.get("speaker", "?")
        text = turn.get("text", "")
        dia_id = turn.get("dia_id", "")
        parts.append(f"[{dia_id}] {speaker}: {text}")
    return "\n".join(parts)


def get_sessions(sample: dict) -> list[tuple[str, str, str]]:
    """Return (session_id, session_text, session_date) for each session."""
    conv = sample["conversation"]
    sessions = []
    keys = sorted(
        [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0,
    )
    for k in keys:
        turns = conv[k]
        if not isinstance(turns, list):
            continue
        date_key = f"{k}_date_time"
        date = conv.get(date_key, "")
        text = build_session_text(turns, date)
        sessions.append((k, text, date))
    return sessions


async def answer_one(client, question, retrieved_memories, sem, model):
    async with sem:
        history = "\n\n".join(
            f"=== Memory {i+1} ===\n{m}"
            for i, m in enumerate(retrieved_memories)
        )
        prompt = f"Retrieved memories:\n\n{history}\n\nQuestion: {question}\n\nAnswer:"
        return await kimi_chat(client, model, QA_SYSTEM, prompt, max_tokens=3000)


async def process_sample(memory, client, sample, top_k, qa_model, sem, use_graph: bool = False):
    """Mem0 ingest sessions, then per-question search + QA."""
    user_id = sample["sample_id"]
    sessions = get_sessions(sample)
    if not sessions:
        return []

    # Wipe per-user memory before ingest (so different conversations don't pollute)
    try:
        memory.delete_all(user_id=user_id)
    except Exception:
        pass

    # Ingest all sessions for this user
    for sid, stext, sdate in sessions:
        try:
            memory.add(stext, user_id=user_id, metadata={"session_id": sid, "date": sdate})
        except Exception as ex:
            print(f"  [mem0 add error {sid}] {ex}", file=sys.stderr)

    qas = sample.get("qa", [])
    results = []

    # Per-question: Mem0 search + Qwen QA
    tasks = []
    for qa in qas:
        question = qa["question"]
        try:
            r = memory.search(question, filters={"user_id": user_id}, limit=top_k)
            mem_items = r.get("results", []) if isinstance(r, dict) else []
            mems = [item.get("memory", "") for item in mem_items[:top_k]]
        except Exception as ex:
            print(f"  [mem0 search error] {ex}", file=sys.stderr)
            mems = []

        async def pack(qa=qa, mems=mems):
            hyp = await answer_one(client, qa["question"], mems, sem, qa_model)
            return qa, mems, hyp

        tasks.append(pack())

    packed = await asyncio.gather(*tasks)
    for qa, mems, hyp in packed:
        results.append({
            "sample_id": sample["sample_id"],
            "question": qa["question"],
            "answer": qa.get("answer", ""),
            "adversarial_answer": qa.get("adversarial_answer", ""),
            "evidence": qa.get("evidence", []),
            "category": qa.get("category"),
            "hypothesis": hyp,
            "retrieved_memories": mems[:3],  # save first 3 for debug
        })
    return results


async def run(data_file: Path, out_file: Path, top_k: int, qa_model: str,
              provider: str, concurrency: int, limit: int, use_graph: bool):
    from mem0 import Memory
    cfg = {}
    if use_graph:
        # Mem0+graph requires Neo4j running; skip for now if no Neo4j configured
        cfg["graph_store"] = {"provider": "neo4j",
                              "config": {"url": "bolt://localhost:7687",
                                         "username": "neo4j",
                                         "password": "password"}}
    memory = Memory.from_config(cfg) if cfg else Memory()
    print(f"[mem0] initialized {'with graph' if use_graph else 'flat'}", file=sys.stderr)

    client = llm_client(provider)
    data = json.load(open(data_file))
    if limit:
        data = data[:limit]
    print(f"[locomo] {len(data)} conversations", file=sys.stderr)

    sem = asyncio.Semaphore(concurrency)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: read existing output to skip done samples
    done_pairs = set()
    if out_file.exists():
        for line in out_file.open():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done_pairs.add((row["sample_id"], row["question"]))
            except Exception:
                pass
    if done_pairs:
        print(f"[resume] skipping {len(done_pairs)} already-done QAs", file=sys.stderr)

    with out_file.open("a") as fout:
        for i, sample in enumerate(data, 1):
            print(f"[sample {i}/{len(data)}] {sample['sample_id']}", file=sys.stderr)
            # Skip if all QAs done
            qas = sample.get("qa", [])
            remaining = [q for q in qas
                         if (sample["sample_id"], q["question"]) not in done_pairs]
            if not remaining:
                print(f"  all {len(qas)} QAs already done, skip", file=sys.stderr)
                continue

            # Filter sample to remaining qa only (to avoid re-processing)
            sample_filtered = dict(sample)
            sample_filtered["qa"] = remaining

            results = await process_sample(memory, client, sample_filtered,
                                            top_k, qa_model, sem, use_graph)
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--provider", type=str, default="qwen")
    ap.add_argument("--qa-model", type=str, default="")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--graph", action="store_true",
                    help="Use Mem0+graph variant (requires Neo4j running)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: set OPENAI_API_KEY (Mem0 uses gpt-4o-mini by default for fact extraction)")

    qa_model = args.qa_model or PROVIDERS[args.provider]["default_model"]
    asyncio.run(run(args.data, args.out, args.top_k, qa_model, args.provider,
                    args.concurrency, args.limit, args.graph))


if __name__ == "__main__":
    main()
